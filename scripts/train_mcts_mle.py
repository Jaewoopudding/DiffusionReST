from collections import defaultdict
import contextlib
import os
import gc
import datetime
from concurrent import futures
import time
import random
import itertools
from absl import app, flags
from ml_collections import config_flags
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import torch.utils
import torch.nn.functional as F
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import tree_pipeline_with_logprob, pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from transformers import CLIPModel
import torchvision
from buffer import PrioritizedReplayBuffer
import warnings
import torch.distributed as dist
warnings.simplefilter(action='ignore', category=FutureWarning)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/svdd_aesthetic_mle.py", "Training configuration.")

logger = get_logger(__name__)


from typing import Optional, Callable, Dict, Any, List, DefaultDict


import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss = mem_info.rss / 1024**2  # Resident Set Size in MB
    vms = mem_info.vms / 1024**2  # Virtual Memory Size in MB
    print(f"[Memory] RSS: {rss:.2f} MB | VMS: {vms:.2f} MB")

# import tracemalloc

# tracemalloc.start()


_CPU_PG = None          # lazy-init 한 번만

def get_cpu_pg():
    global _CPU_PG
    if _CPU_PG is None:
        # 기존 default(NCCL) 그룹은 건드리지 않고 새 그룹 생성
        _CPU_PG = dist.new_group(backend="gloo")
    return _CPU_PG


def cpu_gather_object(obj):
    if not dist.is_initialized():
        return obj                     # single-process run

    pg = get_cpu_pg()
    world_size = dist.get_world_size(pg)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj, group=pg)

    flat = []
    for g in gathered:
        flat.extend(g if isinstance(g, list) else [g])
    return flat



def _split_dict_of_lists(d):
    """
    If d has values that are *all* list/tuple of equal length L,
    turn it into a list[dict] of length L (one element per index).
    Otherwise return [d] untouched.
    """
    list_keys = [k for k, v in d.items() if isinstance(v, (list, tuple))]
    if not list_keys:
        return [d]                      # nothing to split

    L = len(d[list_keys[0]])
    if not all(len(d[k]) == L for k in list_keys):
        return [d]                      # inconsistent → leave as is

    out = []
    for i in range(L):
        new_d = {}
        for k, v in d.items():
            new_d[k] = v[i] if isinstance(v, (list, tuple)) else v
        out.append(new_d)
    return out

def collate_without_tree_prompt(batch):
    """
    batch : list[dict]  (각 dict 구조는
            {'prompt_embeds', 'timesteps', 'latents', ... , 'trees', 'prompts'})

    returns: dict  (same keys; 'trees'·'prompts' 값은 list, 나머지는 batched tensor/array)
    """
    merged = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            merged[k].append(v)

    out = {}
    for k, vlist in merged.items():
        if k in ("trees", "prompts"):
            out[k] = vlist
        else:
            v0 = vlist[0]
            if torch.is_tensor(v0):
                out[k] = torch.cat(vlist, dim=0)
            elif isinstance(v0, np.ndarray):
                out[k] = np.cat(vlist, axis=0)
            else:
                out[k] = vlist
    return out
    

def _flatten_gathered(obj_list):
    """
    • unravel nested lists from gather_object  
    • split dict-of-lists into list-of-dicts
    """
    flat = []
    stack = list(obj_list)              # shallow copy
    while stack:
        itm = stack.pop()
        if isinstance(itm, list):
            stack.extend(itm)           # flatten one level
        elif isinstance(itm, dict):
            flat.extend(_split_dict_of_lists(itm))
        else:
            raise TypeError(f"Unexpected type from gather_object: {type(itm)}")
    return flat

class SearchDataset(torch.utils.data.Dataset):
    """
    Build replay-buffer-style dataset out of on- / off-policy samples.

    Pipeline
    --------
    1.  concatenate on/off, compute global (or per-prompt) percentile
        thresholds for the keys in `filtering_criteria`
    2.  apply **the same filter** to on- and off-policy samples
    3.  keep all filtered on-policy samples (“anchors”)
        + pick exactly `off_policy_subset_size = J` *diverse* off-policy samples
          measured **against those anchors**
          (if J == 0 → skip this step)
    4.  final dataset = anchors ∪ selected off-policy
    """

    # ─────────────────────────────────── init ──────────────────────────────────
    def __init__(
        self,
        on_policy_samples: List[Dict],
        off_policy_samples: List[Dict],
        logger: Optional[Callable],
        *,
        per_prompt_filtering_flag: bool = True,
        per_prompt_select_flag: bool = False,
        filtering_criteria: Dict[str, float],       # e.g. {"reward":0.8,"clip_scores":0.5}
        off_policy_subset_size: int = 0,                   # J (0 → no off-policy extra)
        batch_size_subset: int = 1024,              # B for greedy
        verbose: bool = True,
    ):
        if off_policy_subset_size < 0:
            raise ValueError("off_policy_subset_size (J) must be ≥ 0.")

        self.per_prompt_filtering_flag = per_prompt_filtering_flag
        self.per_prompt_select_flag = per_prompt_select_flag
        self.filtering_criteria = filtering_criteria
        self.logger = logger

        allowed = {"rewards", "clip_scores"}
        bad = set(filtering_criteria) - allowed
        if bad:
            raise ValueError(f"Unknown keys in filtering_criteria: {bad}. Allowed: {allowed}")

        # ── 0) gather all samples ────────────────────────────────────────────
        all_samples = on_policy_samples + off_policy_samples

        # ── 1) thresholds from all samples ──────────────────────────────────
        self.th_global, self.th_per_prompt = self._compute_thresholds(all_samples)

        # ── 2) apply same filter to on & off ────────────────────────────────
        on_pass  = self._apply_filter(on_policy_samples,  self.th_global, self.th_per_prompt)
        off_pass = self._apply_filter(off_policy_samples, self.th_global, self.th_per_prompt)

        if off_policy_subset_size > 0 and off_policy_subset_size > len(off_pass):
            raise ValueError(f"off_policy_subset_size={off_policy_subset_size} but only "
                             f"{len(off_pass)} off-policy candidates after filtering")

        # ── 3) select J diverse off-policy samples w.r.t. on_pass anchors ───
        off_diverse = random.sample(off_pass, off_policy_subset_size) if off_policy_subset_size else []
        
        # off_diverse = []
        # if off_policy_subset_size > 0:
        #     off_diverse = self._select_diverse_candidates(
        #         anchors    = on_pass,
        #         candidates = off_pass,
        #         J          = off_policy_subset_size,
        #         batch_size = batch_size_subset,
        #     )

        # ── 4) merge (anchors ∪ off_diverse) ────────────────────────────────
        self.samples = on_pass + off_diverse    # disjoint by construction

        # ── 5) statistics & report ─────────────────────────────────────────
        if verbose:
            self._calculate_prompts_stats()
            self.report_dataset_stats()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _compute_thresholds(self, samples):
        per_prompt_vals = defaultdict(lambda: defaultdict(list))
        global_vals     = defaultdict(list)

        for s in samples:
            for k, q in self.filtering_criteria.items():
                v = float(s.get(k, np.nan))
                if not np.isnan(v):
                    global_vals[k].append(v)
                    per_prompt_vals[s["prompts"]][k].append(v)

        def q2th(vals, q):
            if not (0 <= q < 1.):
                raise ValueError("filtering_criteria values must be in (0,1)")
            return float(np.quantile(vals, q))

        if self.per_prompt_filtering_flag:
            th_per = {
                p: {k: q2th(per_prompt_vals[p][k], self.filtering_criteria[k])
                    for k in self.filtering_criteria}
                for p in per_prompt_vals
            }
            return None, th_per
        else:
            th_gl = {k: q2th(global_vals[k], self.filtering_criteria[k])
                     for k in self.filtering_criteria}
            return th_gl, None

    def _apply_filter(self, samples, th_global, th_per_prompt):
        if not self.filtering_criteria:
            return samples[:]
        
        if len(samples) == 0:
            return []

        kept = []
        for s in samples:
            th = th_per_prompt[s["prompts"]] if th_per_prompt else th_global
            if all(float(s.get(k, -np.inf)) >= th[k] for k in th):
                kept.append(s)
        return kept

    # ─────────── select J diverse candidates given anchor set ───────────────
    def _select_diverse_candidates(
        self,
        anchors: List[Dict],
        candidates: List[Dict],
        J: int,
        device: Optional[str] = None,
    ) -> List[Dict]:
        """
        k-Center Greedy:  anchors ∪ selected 의 최소거리(= 1-코사인유사도)를
        최대화하는 후보 J개를 반환한다.
        """
        if J == 0 or not candidates:
            return []

        dev = (
            device
            or (anchors[0]["image_embedding"].device if anchors else candidates[0]["image_embedding"].device)
        )

        # --- 임베딩 행렬 준비 ---------------------------------------------------
        def to_matrix(lst):
            if not lst:
                return None
            t = torch.stack([s["image_embedding"].to(dev) for s in lst])
            return F.normalize(t.flatten(1), dim=1)  # [N, D]

        emb_A = to_matrix(anchors)          # [|A|, D]  (None 가능)
        emb_C = to_matrix(candidates)       # [|C|, D]

        N = emb_C.size(0)

        # --- anchors와의 초기 최소거리(1-cosine) 계산 -------------------------
        if emb_A is not None:
            init_sim = torch.matmul(emb_C, emb_A.t())          # cosine similarity
            min_dist = 1 - init_sim.max(dim=1).values          # 1-max-sim → 거리
        else:
            min_dist = torch.ones(N, device=dev)               # anchors 없으면 무한대처럼 시작

        selected_idx = []
        for _ in range(min(J, N)):
            # 1) 현재 최소거리가 가장 큰 후보 선택
            pick = torch.argmax(min_dist).item()
            selected_idx.append(pick)

            # 2) 새로 뽑은 벡터와의 거리를 이용해 min_dist 업데이트
            new_vec = emb_C[pick : pick + 1]                   # [1, D]
            dist_new = 1 - torch.matmul(emb_C, new_vec.t()).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_new)

            # 3) (선택된 인덱스는 다시 뽑히지 않도록) 거리를 -inf로 설정
            min_dist[pick] = -float("inf")

        # ── per-prompt 모드일 경우 재귀 호출로 클래스/프롬프트별 선택 --------
        if self.per_prompt_select_flag:
            chosen_by_prompt = defaultdict(list)
            for idx in selected_idx:
                p = candidates[idx]["prompts"]
                chosen_by_prompt[p].append(idx)

            # 프롬프트별 선택 개수가 부족하면 남은 슬롯을 글로벌로 보충
            deficit = J - len(selected_idx)
            if deficit > 0:
                rest_cands = [c for i, c in enumerate(candidates) if i not in selected_idx]
                selected_idx += [
                    candidates.index(c)
                    for c in self._select_diverse_candidates(anchors, rest_cands, deficit, device=dev)
                ]

        return [candidates[i] for i in selected_idx]

    # ──────────────────────────── statistics ────────────────────────────────
    def _calculate_prompts_stats(self):
        self.prompts_stats = {}
        for s in self.samples:
            p = s["prompts"]
            st = self.prompts_stats.setdefault(p, {"rewards": [], "clip": [], "n": 0})
            st["rewards"].append(float(s["rewards"]))
            st["clip"].append(float(s.get("clip_scores", np.nan)))
            st["n"] += 1
        
    def report_dataset_stats(self):
        self.logger.info("")
        self.logger.info("────────────────── Dataset summary ────────────────────")

        # 1) per-prompt 행별로:  통계 + threshold 함께 출력
        for p, st in self.prompts_stats.items():
            mean_r   = np.nanmean(st["rewards"])
            mean_c   = np.nanmean(st["clip"])
            n        = st["n"]

            # ── 해당 prompt 의 임계값 가져오기 ─────────────────────
            if self.th_per_prompt:
                th = self.th_per_prompt[p]
            else:                          # 글로벌 모드
                th = self.th_global
            th_str = ", ".join(f"{k} ≥ {v:.4f}" for k, v in th.items())

            line = (f"{p[:36]:<36} | R={mean_r:5.3f} | clip={mean_c:5.3f} "
                    f"| n={n:4d} | threshold: {th_str}")
            self.logger.info(line)

        self.logger.info("")
        self.logger.info(f"Filtering criteria (quantile) : {self.filtering_criteria}")
        self.logger.info(f"Final sample count            : {len(self.samples)}")
        self.logger.info("───────────────────────────────────────────────────────\n")


def generate_evaluation_samples(
    pipeline,
    sample_neg_prompt_embeds,
    config,
    accelerator,
    epoch,
    reward_fn,
    executor,
    prompts_history,
    prompts_metadata_history,
    prior_history,
    autocast,
    num_images_per_prompt: int=None
):
    """
    평가용 이미지를 생성하고, log_prob와 reward를 계산하여 eval_samples와 eval_images_list를 반환하는 함수입니다.
    """
    eval_images_list = []
    eval_samples = []
    eval_rewards_list = []
    
    # 평가용 이미지 및 log_prob 샘플링
    for i in tqdm(
        range(config.sample.num_batches_per_epoch) if num_images_per_prompt is None else range(len(prompts_history) * num_images_per_prompt),
        desc=f"Epoch {epoch}: sampling for evaluation",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompt_ids = pipeline.tokenizer(
            prompts_history[i % len(prompts_history)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
        
        with autocast():
            eval_images, _, eval_latents, eval_log_probs = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents=prior_history[i % len(prompts_history)]
            )
            eval_images_list.append(eval_images)
        
        eval_latents = torch.stack(eval_latents, dim=1)
        eval_log_probs = torch.stack(eval_log_probs, dim=1)

        # reward를 비동기로 계산
        eval_rewards_future = executor.submit(reward_fn, eval_images, prompts_history[i % len(prompts_history)], prompts_metadata_history[i % len(prompts_history)])
        time.sleep(0)  # 비동기 호출이 시작될 시간을 주기 위함
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size, 1
        )
        eval_samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": eval_latents[:, :-1],   # 각 timestep 이전의 latent
                "next_latents": eval_latents[:, 1:], # 각 timestep 이후의 latent
                "log_probs": eval_log_probs,
                "eval_rewards": eval_rewards_future,
            }
        )
    
    for sample in tqdm(
        eval_samples,
        desc="Waiting for rewards",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        eval_rewards, _ = sample["eval_rewards"].result()
        sample["eval_rewards"] = torch.as_tensor(eval_rewards, device=accelerator.device)
        eval_rewards_list.append(sample["eval_rewards"])
        eval_results = {
            key: torch.as_tensor(value, device=accelerator.device)
            for key, value in sample.items()
            if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards"]
        }
        sample.update(eval_results)

    eval_samples_collated = {
        k: torch.cat([
            torch.as_tensor(s[k], device=accelerator.device) if isinstance(s[k], np.ndarray) else s[k]
            for s in eval_samples
        ])
        for k in eval_samples[0].keys()
    }

    return eval_samples_collated, eval_images_list, torch.cat(eval_rewards_list)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    
    config.run_name = (
        f'{config.reward_fn}'
        f'_B={config.sample.batch_size * config.sample.num_batches_per_epoch * torch.cuda.device_count()}'
        f'_M={config.search.duplicate}'
        f'_KL={config.train.kl_coef}'
        f'_G={config.search.value_gradient}:{config.search.kl_lagrangian_coef}'
        f'_{datetime.datetime.now().strftime("%Y.%m.%d")}'
        f'_{config.run_name}'
    )
    
    if os.path.exists(os.path.join(config.logdir, config.run_name)):
        for idx in itertools.count(1):
            candidate = f"{config.run_name}_{idx}"
            if not os.path.exists(os.path.join(config.logdir, candidate)):
                config.run_name = candidate
                break

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    def _strip_unpicklable(samples):
        """
        deep-copy 없이 가볍게:  각 샘플 dict에서 'trees' 키를 지운 새 dict 반환
        """
        cleaned = []
        for s in samples:
            d = dict(s)          # 얕은 복사
            d.pop("trees", None)
            cleaned.append(d)
        return cleaned
    

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    if config.multistep_mdp:
        if config.train.type == 'energy_based_negative_gradient':
            accumulation_steps = num_train_timesteps
            if config.train.kl_coef > 0:
                accumulation_steps += num_train_timesteps
            if config.train.negative_gradient:
                accumulation_steps += num_train_timesteps
        else:
            accumulation_steps = num_train_timesteps
    else:
        accumulation_steps = int(config.train.total_batch_size / (torch.cuda.device_count()))
        
    from accelerate.utils import GradientAccumulationPlugin
    plugin = GradientAccumulationPlugin(num_steps=accumulation_steps ,sync_with_dataloader=False)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        # gradient_accumulation_steps=accumulation_steps, # int(config.train.total_batch_size / (torch.cuda.device_count()))
        gradient_accumulation_plugin=plugin
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ddpo-pytorch",
            config=config.to_dict(),
            init_kwargs={
                "wandb": 
                {
                    "name": config.run_name,
                    "entity": "gda-for-orl",
                }
            },
        )
    logger.info(f"\n{config}")



    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not re ired.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    
    
    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        # assert len(models) == 1
        try:
            if config.use_lora and isinstance(models[0], AttnProcsLayers):
                pipeline.unet.save_attn_procs(output_dir)
            elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
                models[0].save_pretrained(os.path.join(output_dir, "unet"))
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            weights.pop()  # ensures that accelerate doesn't try to handle saving of the model
        except:
            print("Error occurred while saving model")

    def load_model_hook(models, input_dir):
        # assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model,
                revision=config.pretrained.revision,
                subfolder="unet",
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(
                AttnProcsLayers(tmp_unet.attn_processors).state_dict()
            )
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    prompts_total, prompt_metadata = prompt_fn(**config.prompt_fn_kwargs)
    num_prompts = len(prompts_total)
    
    prior_total_for_eval = [torch.randn(config.sample.batch_size, 4, 64, 64).to(accelerator.device) * pipeline.scheduler.init_noise_sigma for _ in range(len(prompts_total))]
    prompt_metadata_total_for_eval = [{} for _ in range(len(prompts_total))] 
    
    # if "aesthetic" in config.reward_fn :
    #     reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)(torch_dtype=inference_dtype)
    # else:
    #     reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    # eval_fn = getattr(ddpo_pytorch.rewards, config.eval_fn)()
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.per_prompt_stat_tracking.buffer_size,
            config.per_prompt_stat_tracking.min_count,
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    resizer = torchvision.transforms.Resize(224)
    normalizer = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    if config.train.kl_coef > 0 or config.train.type == 'dpo':

        unet_pretrained = UNet2DConditionModel.from_pretrained(
            config.pretrained.model,
            revision=config.pretrained.revision,
            subfolder="unet",
        ).to(accelerator.device, dtype=inference_dtype)

        # Prepare everything with our `accelerator`.
        unet, optimizer, unet_pretrained = accelerator.prepare(unet, optimizer, unet_pretrained)
        
    else:
        unet, optimizer = accelerator.prepare(unet, optimizer)
        

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = (
        config.sample.batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info("")
    logger.info(f"  Gradient steps per single improve step = {config.train.gradient_steps_per_improve_step}")
    logger.info(f"  Total number of samples per grow step = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {config.train.total_batch_size}"
    )
    logger.info(f"  Number of improve steps per grow step = {config.train.improve_steps}")
    logger.info(f"  Frequency for save = {config.save_freq}")
    logger.info(f"  Kullback-Liebler divergence coefficient = {config.train.kl_coef}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    # assert config.train.total_batch_size % accelerator.num_processes == 0
    # assert config.eval.num_images_per_prompt % accelerator.num_processes == 0, "eval.num_prompts_per_batch must be devided  for now"


    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    
    global_step = 0
    off_policy_dataset_per_gpu = []
    
    
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        on_policy_dataset_per_gpu = []
        buffers = [PrioritizedReplayBuffer(capacity=10000, priority="rewards") for _ in range(num_prompts)]
        pipeline.unet.eval()
        samples = []
        eval_samples = []
        prompts = []
        images_list = []
        eval_images_list = []
        num_images_per_prompt = config.sample.num_batches_per_epoch // config.sample.num_prompts_per_batch
        
        prompts_history = []
        prompts_metadata_history = []
        prior_history = []
        
        idxs = np.random.choice(num_prompts, size=config.sample.num_prompts_per_batch, replace=False).tolist()
        prompts = [prompts_total[i] for i in idxs]  

        image_embedder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        image_embedder.to(accelerator.device, dtype=inference_dtype)
        
        
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            prompt = prompts[i // num_images_per_prompt]
            
            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                shape = (config.search.duplicate * config.search.nfe_per_action, 4, 64, 64)
                init_latents = torch.randn(shape, device=accelerator.device)
                pipeline.batch_size = 1
                images, _, latents, log_probs, prior, tree = tree_pipeline_with_logprob(
                    pipeline,
                    config=config,
                    reward_fn=reward_fn,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    latents=init_latents,
                    prompts=prompt,
                    prompt_metadata=prompt_metadata,
                    ref_unet = unet_pretrained if config.search.importance_sampling else None,
                )
                images_list.append(images)
                prior_history.append(prior)
                prompts_history.append(prompt)
                prompts_metadata_history.append(prompt_metadata)

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)
            

            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)

            time.sleep(0)
            
            processed_images = normalizer(resizer(images)).to(inference_dtype)
            with torch.no_grad():
                if isinstance(image_embedder, torch.nn.parallel.distributed.DistributedDataParallel):
                    embedded_images = image_embedder.module.get_image_features(pixel_values=processed_images)
                    embedded_prompts = image_embedder.module.get_text_features(input_ids=prompt_ids)   
                else:
                    embedded_images = image_embedder.get_image_features(pixel_values=processed_images)  
                    embedded_prompts = image_embedder.get_text_features(input_ids=prompt_ids)
            embedded_images = (embedded_images / torch.norm(embedded_images, dim=-1, keepdim=True)).detach()
            embedded_prompts = (embedded_prompts / torch.norm(embedded_prompts, dim=-1, keepdim=True)).detach()
            logits_per_image = embedded_images @ embedded_prompts.T
            clip_scores = torch.diagonal(logits_per_image)
        

            samples.append(
                {
                    "prompts": prompt,
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    # "evals": evals,
                    "final_embedding": latents[:, -1],
                    "tree": tree,
                    "image_embedding": embedded_images,
                    "clip_scores": clip_scores,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)
            # eval_results = sample["evals"].result()
            eval_results = {key: torch.as_tensor(value, device=accelerator.device) for key, value in sample.items() if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards", "tree", "prompts"]}
            sample.update(eval_results)
            # del sample["evals"]

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        trees = [sample['tree'] for sample in samples]
        prompts_list = [sample["prompts"] for sample in samples]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()if k not in ['tree', 'prompts']}
        samples['trees'] = trees
        samples['prompts'] = prompts_list
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        del image_embedder
        gc.collect()
        torch.cuda.empty_cache()
        
        save_dir = f'images/{config.run_name}'
        search_dir = os.path.join(save_dir, f"search_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True) 
        os.makedirs(search_dir, exist_ok=True) 
        rank = accelerator.process_index  # 0, 1, ...
        for i, (image, prompt) in enumerate(zip(images_list, prompts_history)):
            filename = (
                f"G{epoch+1}_rank{rank}_idx{i}"
                f"_{prompt[:40].replace(os.sep,'_')}"          # (경로 구분자 무력화)
                f"_{samples['rewards'][i]:.4f}.jpg"
            )
            pil_img = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil_img.save(os.path.join(search_dir, filename))
        
        if dist.is_initialized():
            accelerator.wait_for_everyone()
        # for i, (image, prompt) in enumerate(zip(images_list, prompts_history)):
        #     pil = Image.fromarray(
        #         (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        #     )
        #     pil.save(os.path.join(search_dir, f"G:{epoch+1}_{prompt}_search_{(i + 1) * (accelerator.local_process_index + 1)}_{samples['rewards'][i]:.4f}.jpg"))


        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        clip_scores = accelerator.gather(samples["clip_scores"]).cpu().numpy()
        # metrics = {key: accelerator.gather(torch.stack([s[key] for s in samples])).cpu().numpy() for key in eval_results.keys()}
        
        log_dict = {
            "reward": rewards,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "clip_score": clip_scores,
            "clip_score_mean": clip_scores.mean(),
            "clip_score_std": clip_scores.std(),
        }

        accelerator.log(log_dict, step=global_step)

        eval_samples, eval_images_list, eval_rewards = generate_evaluation_samples(
            pipeline=pipeline,
            sample_neg_prompt_embeds=sample_neg_prompt_embeds,
            config=config,
            accelerator=accelerator,
            epoch=epoch,
            reward_fn=reward_fn,
            executor=executor,
            prompts_history=prompts_history,
            prompts_metadata_history=prompts_metadata_history,
            prior_history=prior_history,
            autocast=autocast
        )
        
        samples['eval_latents'] = eval_samples['latents']
        samples['eval_next_latents'] = eval_samples['next_latents']
        samples['eval_log_probs'] = eval_samples['log_probs']

        if epoch == 0:
            for i, image in enumerate(eval_images_list):
                pil = Image.fromarray(
                    (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(save_dir, f"{epoch}_{(i + 1) * (accelerator.local_process_index + 1)}_eval_{eval_rewards[i]:.4f}.jpg"))
                
            # with tempfile.TemporaryDirectory() as tmpdir:
            #     for i, image in enumerate(eval_images_list[0]):
            #         pil = Image.fromarray(
            #             (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            #         )
            #         pil = pil.resize((256, 256))
            #         pil.save(os.path.join(tmpdir, f"{i}_eval.jpg"))
                    
            #     accelerator.log(
            #         {
            #             "eval_images": [
            #                 wandb.Image(
            #                     os.path.join(tmpdir, f"{i}_eval.jpg"),
            #                     caption=f"{prompt:.25} | {eval_reward:.2f}",
            #                 )
            #                 for i, (prompt, eval_reward) in enumerate(
            #                     zip(prompts, eval_rewards)
            #                 )  # only log rewards from process 0
            #             ],
            #         },
            #         step=global_step,
            #     )

            # log rewards and images
            log_dict = {
                "eval_reward": eval_rewards,
                "eval_reward_mean": eval_rewards.mean(),
                "eval_reward_std": eval_rewards.std(),
            }

            accelerator.log(log_dict, step=global_step)


        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, rewards)
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        # shuffle along time dimension independently for each sample
        total_batch_size, num_timesteps = samples["timesteps"].shape
        perms = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])

        for key in ["timesteps", "latents", "next_latents", "log_probs", "eval_latents", "eval_next_latents", "eval_log_probs"]:
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=accelerator.device)[:, None],
                perms,
            ]


        samples_batched = {
            k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) if k not in ['trees', 'prompts'] else v
            for k, v in samples.items()
        }

        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]
        
        for i, sample in enumerate(samples_batched):
            buffers[idxs[i // num_images_per_prompt]].push(sample)
        
        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps
        
        def _to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            elif isinstance(x, dict):
                return {k: _to_cpu(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(_to_cpu(v) for v in x)
            else:
                return x


        ## SAVE 하기 전, rewards랑 clip_score, 그리고 embedding을 따로 저장.
        ## 따로 저장한 rewards, clipscore, embedding을 불러오고, filtering 대상을 각 gpu process에서 제거함.
        ## 제거한 데이터를 제외한 나머지 데이터를 buffer_path에 저장함
        
        samples_batched_cpu = [_to_cpu(s) for s in samples_batched]
        for i, sample in enumerate(samples_batched_cpu):
            on_policy_dataset_per_gpu.append(sample)

        on_policy_dataset_per_gpu  = _strip_unpicklable(on_policy_dataset_per_gpu)
        off_policy_dataset_per_gpu = _strip_unpicklable(off_policy_dataset_per_gpu)
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        buffer_dir = os.path.join("buffer", config.run_name, f"epoch{epoch+1}")
        os.makedirs(buffer_dir, exist_ok=True)
        buffer_path = os.path.join(buffer_dir, f"buffer_{rank}.pt")
        torch.save({
            "on_policy_dataset": on_policy_dataset_per_gpu,
            "off_policy_dataset": off_policy_dataset_per_gpu,
        }, buffer_path)
        
        
        if dist.is_initialized():
            accelerator.wait_for_everyone()
        
        
        on_policy_dataset = []
        off_policy_dataset = []
        for r in range(world_size):
            path = os.path.join(buffer_dir, f"buffer_{r}.pt")
            data = torch.load(path)
            on_policy_dataset.extend(data["on_policy_dataset"])
            off_policy_dataset.extend(data["off_policy_dataset"])

        ## Dataset은 on_policy_dataset과 off_policy_dataset list를 받는 게 아니라 buffer의 경로를 받아옴.
        ## 요청이 들어올 때마다 데이터를 읽고 반환함. 메모리 절약을 위해서임. 
        ## 그에 맞게 SearchDataset 코드를 수정하는 방안을 제시할 것.
        
        replaybuffer = SearchDataset(
            on_policy_samples   = on_policy_dataset,
            off_policy_samples  = off_policy_dataset,
            per_prompt_filtering_flag = config.buffer.per_prompt_filtering_flag,
            per_prompt_select_flag = config.buffer.per_prompt_select_flag,
            filtering_criteria  = {"rewards": config.buffer.reward_filtering_criteria, "clip_scores": config.buffer.clip_score_filtering_criteria},
            off_policy_subset_size     = 0 if len(off_policy_dataset) == 0 else config.buffer.off_policy_subset_size, # off policy sample 중 prompt당 3개씩 가져오기. 
            batch_size_subset   = 2048,
            logger              = logger
        )

        if config.buffer.off_policy_subset_size > 0:
            for i, sample in enumerate(samples_batched_cpu):
                off_policy_dataset_per_gpu.append(sample)

        dataloader = torch.utils.data.DataLoader(
            replaybuffer,
            batch_size=config.train.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            collate_fn=collate_without_tree_prompt,
        )

        dataloader = accelerator.prepare_data_loader(dataloader)    
        on_policy_dataset_per_gpu.clear()
        # all_on_policy_samples.clear()
        # all_off_policy_samples.clear()

        # del all_on_policy_samples
        # del all_off_policy_samples
        del samples_batched_cpu
        del samples_batched
        del samples
        del trees
        
        gc.collect()
        torch.cuda.empty_cache()    
        time.sleep(1)
    
        #################### TRAINING ####################
        
        for improve_steps in range(config.train.improve_steps):
            # train
            pipeline.unet.train()
            mse_loss = torch.nn.MSELoss()
            info = defaultdict(list)
            
            if config.multistep_mdp:
                if config.train.type == 'awac':        
                    for i, sample in tqdm(
                        list(enumerate(samples_batched)),
                        desc=f"Epoch {epoch}.{improve_steps}: training",
                        position=0,
                        disable=not accelerator.is_local_main_process,
                    ):
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                        search_tree = sample['trees']
                        search_tree.reset_root_nodes()
                        
                        for j in tqdm(
                            range(num_train_timesteps),
                            desc="Timestep",
                            position=1,
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                        ):
                            search_tree.act_and_prune(tree.argmax_value, prune=False)
                            with accelerator.accumulate(unet):
                                with autocast():
                                    current_nodes = search_tree.root_nodes
                                    latents = current_nodes.states
                                    timesteps = current_nodes.timesteps
                                    
                                    noise_pred = unet(torch.cat([latents] * 2),torch.cat([timesteps] * 2),embeds,).sample
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.sample.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    )
                                    
                                    child_rewards = torch.tensor([child.reward for child in current_nodes.get_children()[0]], dtype=torch.float32, device=accelerator.device).view(-1, 1)
                                    advantages = torch.exp((child_rewards - child_rewards.mean()) / (child_rewards.std() + 1e-8))
                                advantages = torch.clamp(advantages,-config.train.adv_clip_max, config.train.adv_clip_max )
                                for idx, child in enumerate(current_nodes.get_children()[0]):
                                    _, log_prob = ddim_step_with_logprob(
                                        pipeline.scheduler,
                                        noise_pred,
                                        timesteps.to(torch.int64),
                                        latents,
                                        eta=config.sample.eta,
                                        prev_sample=child.state
                                    )
                                    loss = - advantages[idx] * log_prob
                                    info['loss'].append(loss)
                                    if idx ==  len(current_nodes.get_children()[0]) - 1:
                                        accelerator.backward(loss)
                                    else:
                                        accelerator.backward(loss, retain_graph=True)
                                    if accelerator.sync_gradients:
                                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                        info = accelerator.reduce(info, reduction="mean")
                                        info.update({"epoch": epoch, "improve_steps": improve_steps})
                                        accelerator.log(info, step=global_step)
                                        global_step += 1
                                        info = defaultdict(list)
                                        accelerator.clip_grad_norm_(
                                            unet.parameters(), config.train.max_grad_norm
                                        )
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    
                elif config.train.type == 'energy_based_negative_gradient': 
                    for sample in tqdm(
                        dataloader,
                        desc=f"Epoch {epoch}.{improve_steps}: training",
                        position=0,
                        disable=not accelerator.is_local_main_process,
                    ):
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                        
                        for j in tqdm(
                            range(num_train_timesteps),
                            desc="Timestep",
                            position=1,
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                        ):
                            with accelerator.accumulate(unet):
                                with autocast():
                                    # print_memory_usage()
                                    
                                    # snapshot = tracemalloc.take_snapshot()
                                    # top_stats = snapshot.statistics('lineno')

                                    # print("[ Top 5 memory consumers ]")
                                    # for stat in top_stats[:5]:
                                    #     print(stat)

                                    latents = sample['latents'][:, j]
                                    timesteps = sample['timesteps'][:, j]
                                    next_latents = sample['next_latents'][:, j]
                                    
                                    noise_pred = unet(torch.cat([latents] * 2),torch.cat([timesteps] * 2), embeds,).sample
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.sample.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    )
                                    _, search_log_prob = ddim_step_with_logprob(
                                        pipeline.scheduler,
                                        noise_pred,
                                        timesteps.to(torch.int64),
                                        latents,
                                        eta=config.sample.eta,
                                        prev_sample=next_latents
                                    )
                                    
                                loss = -search_log_prob
                                info["positive_loss"].append(loss.detach())
                                accelerator.backward(loss)
                                if (config.train.kl_coef) == 0 and (not config.train.negative_gradient) and (accelerator.sync_gradients):
                                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                    info = accelerator.reduce(info, reduction="mean")
                                    info.update({"epoch": epoch, "improve_steps": improve_steps})
                                    accelerator.log(info, step=global_step)
                                    global_step += 1
                                    info = defaultdict(list)
                                    accelerator.clip_grad_norm_(
                                        unet.parameters(), config.train.max_grad_norm
                                    )
                                optimizer.step()
                                optimizer.zero_grad()

                            if config.train.kl_coef > 0:
                                with accelerator.accumulate(unet):
                                    with autocast():
                                        noise_pred = unet(
                                            torch.cat([latents] * 2),
                                            torch.cat([timesteps] * 2), 
                                            embeds,
                                        ).sample
                                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                        noise_pred = (
                                            noise_pred_uncond
                                            + config.sample.guidance_scale
                                            * (noise_pred_text - noise_pred_uncond)
                                        )
                                        with torch.no_grad():
                                            ref_noise_pred = unet_pretrained(
                                                torch.cat([latents] * 2),
                                                torch.cat([timesteps] * 2),
                                                embeds,
                                            ).sample
                                            ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
                                            ref_noise_pred = (
                                                ref_noise_pred_uncond
                                                + config.sample.guidance_scale
                                                * (ref_noise_pred_text - ref_noise_pred_uncond)
                                            )
                                    kl_loss = config.train.kl_coef * mse_loss(noise_pred, ref_noise_pred.detach())
                                    info["kl_loss"].append(kl_loss.detach())
                                    accelerator.backward(kl_loss)
                                    if (not config.train.negative_gradient) and (accelerator.sync_gradients):
                                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                        info = accelerator.reduce(info, reduction="mean")
                                        info.update({"epoch": epoch, "improve_steps": improve_steps})
                                        accelerator.log(info, step=global_step)
                                        global_step += 1
                                        info = defaultdict(list)
                                        accelerator.clip_grad_norm_(
                                            unet.parameters(), config.train.max_grad_norm
                                        )
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    
                            if config.train.negative_gradient:
                                with accelerator.accumulate(unet):
                                    with autocast():
                                        latents = sample['eval_latents'][:, j]
                                        next_latents = sample['eval_next_latents'][:, j]
                                        neg_noise_pred = unet(torch.cat([latents] * 2), torch.cat([timesteps] * 2), embeds,).sample
                                        neg_noise_pred_uncond, neg_noise_pred_text = neg_noise_pred.chunk(2)
                                        neg_noise_pred = (
                                            neg_noise_pred_uncond
                                            + config.sample.guidance_scale
                                            * (neg_noise_pred_text - neg_noise_pred_uncond)
                                        )
                                        _, neg_log_prob = ddim_step_with_logprob(
                                            pipeline.scheduler,
                                            neg_noise_pred,
                                            timesteps.to(torch.int64),
                                            latents,
                                            eta=config.sample.eta,
                                            prev_sample=next_latents,
                                        )         

                                    neg_prob_ref = sample['eval_log_probs'][:, j]
                                    neg_prob_threshold = neg_prob_ref * (1 - torch.sign(neg_prob_ref) * config.train.clip_range)
                                    loss = torch.clip(neg_log_prob, min=neg_prob_threshold)
                                    info["negative_loss"].append(loss.detach())
                                    info["clipfrac"].append(
                                        torch.mean(
                                            (
                                                neg_log_prob < neg_prob_threshold
                                            ).float().view(-1, 1)
                                        )
                                    )
                                    
                                    accelerator.backward(loss)
                                    if accelerator.sync_gradients:
                                        # log training-related stuff
                                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                        info = accelerator.reduce(info, reduction="mean")
                                        info.update({"epoch": epoch, "improve_steps": improve_steps})
                                        
                                        accelerator.log(info, step=global_step)
                                        global_step += 1
                                        info = defaultdict(list)
                                        accelerator.clip_grad_norm_(
                                            unet.parameters(), config.train.max_grad_norm
                                        )
                                    optimizer.step()
                                    optimizer.zero_grad()
                                
                elif config.train.type == 'dpo':
                    for i, sample in tqdm(
                        list(enumerate(samples_batched)),
                        desc=f"Epoch {epoch}.{improve_steps}: training",
                        position=0,
                        disable=not accelerator.is_local_main_process,
                    ):
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                        
                        for j in tqdm(
                            range(num_train_timesteps),
                            desc="Timestep",
                            position=1,
                            leave=False,
                            disable=not accelerator.is_local_main_process,
                        ):
                            with accelerator.accumulate(unet):
                                with autocast():
                                    latents = sample['latents'][:, j]
                                    timesteps = sample['timesteps'][:, j]
                                    next_latents = sample['next_latents'][:, j]
                                    
                                    noise_pred = unet(torch.cat([latents] * 2),torch.cat([timesteps] * 2), embeds,).sample
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.sample.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond)
                                    )
                                    _, search_log_prob = ddim_step_with_logprob(
                                        pipeline.scheduler,
                                        noise_pred,
                                        timesteps.to(torch.int64),
                                        latents,
                                        eta=config.sample.eta,
                                        prev_sample=next_latents
                                    )
                                    
                                    latents = sample['eval_latents'][:, j]
                                    next_latents = sample['eval_next_latents'][:, j]
                                    neg_noise_pred = unet(torch.cat([latents] * 2), torch.cat([timesteps] * 2), embeds,).sample
                                    neg_noise_pred_uncond, neg_noise_pred_text = neg_noise_pred.chunk(2)
                                    neg_noise_pred = (
                                        neg_noise_pred_uncond
                                        + config.sample.guidance_scale
                                        * (neg_noise_pred_text - neg_noise_pred_uncond)
                                    )
                                    _, neg_log_prob = ddim_step_with_logprob(
                                        pipeline.scheduler,
                                        neg_noise_pred,
                                        timesteps.to(torch.int64),
                                        latents,
                                        eta=config.sample.eta,
                                        prev_sample=next_latents,
                                    )         
                                loss = torch.nn.functional.binary_cross_entropy_with_logits(config.train.beta_dpo * (search_log_prob - neg_log_prob)).mean()                  
                                accelerator.backward(loss)
                                if accelerator.sync_gradients:
                                    # log training-related stuff
                                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                    info = accelerator.reduce(info, reduction="mean")
                                    info.update({"epoch": epoch, "improve_steps": improve_steps})
                                    accelerator.log(info, step=global_step)
                                    global_step += 1
                                    info = defaultdict(list)
                                    accelerator.clip_grad_norm_(
                                        unet.parameters(), config.train.max_grad_norm
                                    )
                                optimizer.step()
                                optimizer.zero_grad()
            else:
                ################################################
                # 샘플 잔뜩 뽑고, buffer에서 뽑아서 사용하는 방식을 채택한다.
                ################################################
                for step in tqdm(
                    range(config.train.gradient_steps_per_improve_step),
                    desc=f"Grow: {epoch + 1} | Improve {improve_steps + 1} | Total gradient steps {config.train.gradient_steps_per_improve_step} ",
                    position=0,
                    leave=True,
                    disable=not accelerator.is_local_main_process,
                ):
                    possible_idxs = []
                    for i in range(num_prompts):
                        if len(buffers[i].buffer) > 0:
                            possible_idxs.append(i)
                    samples_from_buffer = []
                    for _ in range(int(config.train.total_batch_size / (accelerator.num_processes))):
                        buffer = buffers[random.choice(possible_idxs)]
                        threshold = -1e3 if config.train.type == 'dpo' else buffer.reward_median()
                        samples_from_buffer.extend(buffer.sample(1, target_threshold={"rewards": threshold}))
                        
                    for j, sample in enumerate(tqdm(
                        samples_from_buffer,
                        position=1,
                        leave=False,
                        desc=f'Grow: {epoch + 1} | Improve {improve_steps + 1} | Batch Iteration',
                        disable=not accelerator.is_local_main_process,)
                    ):
                        with accelerator.accumulate(unet):
                            with autocast():
                                if config.train.type == 'sft':
                                    if config.train.cfg:
                                        # concat negative prompts to sample prompts to avoid two forward passes
                                        embeds = torch.cat(
                                            [train_neg_prompt_embeds, sample["prompt_embeds"]]
                                        )
                                    else:
                                        embeds = sample["prompt_embeds"]
                                
                                    if config.train.cfg:
                                        clean_latents = sample["latents"][:, -1]
                                        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (config.train.batch_size,), device=accelerator.device)
                                        noise = torch.randn_like(clean_latents)
                                        noised_latents = pipeline.scheduler.add_noise(clean_latents, noise, timesteps)

                                        noise_pred = unet(
                                            torch.cat([noised_latents] * 2),
                                            torch.cat([timesteps] * 2),
                                            embeds,
                                        ).sample
                                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                        noise_pred = (
                                            noise_pred_uncond
                                            + config.sample.guidance_scale
                                            * (noise_pred_text - noise_pred_uncond)
                                        )
                                        
                                        if config.train.kl_coef > 0:
                                            with torch.no_grad():
                                                ref_noise_pred = unet_pretrained(
                                                    torch.cat([noised_latents] * 2),
                                                    torch.cat([timesteps] * 2),
                                                    embeds,
                                                ).sample
                                                ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
                                                ref_noise_pred = (
                                                    ref_noise_pred_uncond
                                                    + config.sample.guidance_scale
                                                    * (ref_noise_pred_text - ref_noise_pred_uncond)
                                                )
                                    else:
                                        raise NotImplementedError("Not implemented yet")
                                    loss = mse_loss(noise_pred, noise)
                                    
                                    if config.train.kl_coef > 0:
                                        kl_loss = config.train.kl_coef * mse_loss(noise_pred, ref_noise_pred.detach())
                                        loss = loss + kl_loss
                                        info["kl_loss"].append(kl_loss)
                                    info["loss"].append(loss)
                                    
                                elif config.train.type == 'dpo':
                                    clean_latents = torch.cat([sample["latents"][:, -1], sample["eval_latents"][:, -1]])
                                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (config.train.batch_size,), device=accelerator.device)
                                    timesteps = timesteps.long().chunk(2)[0].repeat(2)
                                    noise = torch.randn_like(clean_latents)
                                    noised_latents = pipeline.scheduler.add_noise(clean_latents, noise, timesteps)
                                    embeds = sample["prompt_embeds"].repeat(2, 1, 1)
                                    
                                    model_pred = unet(clean_latents, timesteps, embeds).sample
                                    model_losses = (model_pred - noise).pow(2).mean(dim=[1,2,3])
                                    model_losses_w, model_losses_l = model_losses.chunk(2)
                                    
                                    raw_model_loss = (model_losses_w.mean() - model_losses_l.mean())
                                    model_diff = model_losses_w - model_losses_l
                                    
                                    with torch.no_grad():
                                        ref_noise_pred = unet_pretrained(clean_latents, timesteps, embeds).sample
                                        ref_losses = (ref_noise_pred - noise).pow(2).mean(dim=[1,2,3])
                                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                                        ref_diff = ref_losses_w - ref_losses_l
                                        raw_ref_loss = ref_losses.mean()
                                        
                                    scale_term = -0.5 * config.train.beta_dpo
                                    inside_term = scale_term * (model_diff - ref_diff)
                                    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                                    loss = -1 * torch.nn.functional.logsigmoid(inside_term).mean()
                                    
                                    avg_loss = accelerator.gather(loss.repeat(config.train.total_batch_size)).mean()
                                    info["avg_loss"].append(avg_loss)
                                    info["train_loss"].append(avg_loss / int(config.train.total_batch_size / (torch.cuda.device_count())))
                                    info["avg_model_mse"].append(accelerator.gather(raw_model_loss.repeat(config.train.total_batch_size)).mean())
                                    info["avg_ref_mse"].append(accelerator.gather(raw_ref_loss.repeat(config.train.total_batch_size)).mean())
                                    info["avg_acc"].append(accelerator.gather(implicit_acc).mean())
                                else:
                                    raise ValueError("Unknown training type")
                                    
                            
                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                assert j == len(samples_from_buffer) - 1
                                # log training-related stuff
                                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                info = accelerator.reduce(info, reduction="mean")
                                info.update({"epoch": epoch, "improve_steps": improve_steps})
                                accelerator.log(info, step=global_step)
                                global_step += 1
                                info = defaultdict(list)
                                accelerator.clip_grad_norm_(
                                    unet.parameters(), config.train.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()
                        
                            # if accelerator.sync_gradients:
                                # assert (int(config.train.gradient_steps_per_improve_step / (accelerator.num_processes * config.train.batch_size))) % config.train.gradient_accumulation_steps == 0
            

            eval_samples, eval_images_list, eval_rewards = generate_evaluation_samples(
                pipeline=pipeline,
                sample_neg_prompt_embeds=sample_neg_prompt_embeds,
                config=config,
                accelerator=accelerator,
                epoch=epoch,
                reward_fn=reward_fn,
                executor=executor,
                prompts_history=prompts_total,
                prompts_metadata_history=prompt_metadata_total_for_eval,
                prior_history=prior_total_for_eval,
                autocast=autocast,
                num_images_per_prompt=1 # config.eval.num_images_per_prompt // accelerator.num_processes,
            )

            eval_images_tensor = torch.cat(eval_images_list)
            
            eval_dir = os.path.join(save_dir, f"eval_{epoch+1}-improve_{improve_steps+1}")
            os.makedirs(eval_dir, exist_ok=True)
            rank = accelerator.process_index
            for i, (image, prompt) in enumerate(zip(eval_images_tensor, prompts_total)):
                filename = (
                    f"G{epoch+1}_rank{rank}_idx{i}"
                    f"_{prompt[:40].replace(os.sep,'_')}"
                    f"_{eval_rewards[i]:.4f}.jpg"
                )
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(eval_dir, filename))
            if dist.is_initialized():
                accelerator.wait_for_everyone()
    
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_images = eval_images_list[0]  # 명확하게 지정
                for i, image in enumerate(eval_images):
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}_eval.jpg"))
                
                accelerator.log(
                    {
                        "eval_images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{i}_eval.jpg"),
                                caption=f"{prompt:.25} | {eval_reward:.2f}",
                            )
                            for i, (prompt, eval_reward) in enumerate(
                                zip(prompts[:len(eval_images_list[0])], eval_rewards[:len(eval_images_list[0])])
                            )
                        ],
                    },
                    step=global_step,
                )

            eval_rewards = accelerator.gather(eval_rewards).cpu().numpy()
            # log rewards and images
            log_dict = {
                "eval_reward": eval_rewards,
                "eval_reward_mean": eval_rewards.mean(),
                "eval_reward_std": eval_rewards.std(),
            }
            accelerator.log(log_dict, step=global_step)


            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            accelerator._dataloaders.clear()
            del replaybuffer
            del dataloader
            del eval_samples
            del eval_images_list
            del eval_rewards
            del eval_images_tensor
            
            
            gc.collect()
            torch.cuda.empty_cache()    

        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()




if __name__ == "__main__":
    app.run(main)
