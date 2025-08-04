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
from pathlib import Path
import re 
warnings.simplefilter(action='ignore', category=FutureWarning)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


from typing import Optional, Callable, Dict, Any, List, DefaultDict


import psutil


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
    


class SearchDataset(torch.utils.data.Dataset):
    """
    데이터는 (1) buffer/epoch_{E}_rank_{R}.pt 에 저장된 샘플 리스트
            (2) meta_epoch_{E}_rank_{R}.pt 에 저장된 메타-정보
    두 파일의 *순서가 동일* 하다는 가정하에 lazy-loading 으로 동작한다.
    """

    _FILE_RGX = re.compile(r"epoch_(\d+)_rank_(\d+)\.pt")

    # ─────────────────────────────────── init ─────────────────────────────────
    def __init__(
        self,
        metadata: Dict[str, List],        # gathered_meta   (length = N_total)
        buffer_path: str | Path,          # "buffer" 디렉터리
        logger,
        epoch: int,
        *,
        per_prompt_filtering_flag: bool = True,
        per_prompt_select_flag: bool = False,
        filtering_criteria: Dict[str, float] = None,
        off_policy_subset_size: int = 0,
        batch_size_subset: int = 1024,
        verbose: bool = True,
    ):
        self.logger = logger
        self.per_prompt_filtering_flag = per_prompt_filtering_flag
        self.per_prompt_select_flag   = per_prompt_select_flag
        self.filtering_criteria       = filtering_criteria or {}
        self.off_policy_subset_size   = off_policy_subset_size
        self.batch_size_subset        = batch_size_subset

        # -------------------------- 0. 메타 정보 ------------------------------
        self.meta = metadata                    # {prompts, rewards, clip_scores, image_embedding}
        N = len(self.meta["prompts"])
        assert all(len(v) == N for v in self.meta.values()), "메타 각 항목 길이가 다름"

        # -------------------------- 1. 인덱스 매핑 ----------------------------
        #   global idx  →  (epoch, rank, local_idx)
        #   파일별 샘플 수는 한번만 읽어 길이를 캐시
        self.buffer_root = Path(buffer_path)
        self._file_len_cache: Dict[tuple[int,int], int] = {}
        self._file_data_cache: Dict[tuple[int,int], List[dict]] = {}
        self.current_epoch = epoch

        # gather order와 동일한 (epoch,rank) 순으로 mapping 을 만든다
        index_map: list[tuple[int,int,int]] = []
        cursor = 0
        for meta_file in sorted(self.buffer_root.glob("meta_epoch_*_rank_*.pt"),
                                key=lambda p: (int(p.stem.split("_")[2]), int(p.stem.split("_")[-1]))):
            m = self._FILE_RGX.search(meta_file.name)
            epoch, rank = int(m.group(1)), int(m.group(2))
            ds_file = self.buffer_root / f"epoch_{epoch}_rank_{rank}.pt"
            # 샘플 수(cache)
            if (epoch, rank) not in self._file_len_cache:
                self._file_len_cache[(epoch, rank)] = len(torch.load(ds_file, map_location="cpu")["dataset"])
            L = self._file_len_cache[(epoch, rank)]
            for i in range(L):
                index_map.append((epoch, rank, i))
                cursor += 1
        
        assert cursor == N, "메타 길이와 실제 파일의 샘플 수가 불일치합니다."

        self._index_map = index_map                         # 전체 길이 = N

        # -------------------------- 2. Threshold 계산 -------------------------
        self._compute_and_apply_filter(verbose)
        
        
        self._calculate_prompts_stats()          # kept-샘플 통계 만들기
        if verbose:
            self.report_dataset_stats()          # 화면에 보기 좋게 출력

    # ──────────────────────── helper : threshold & filter ────────────────────
    def _compute_and_apply_filter(self, verbose: bool):
        """self.meta 로부터 threshold 계산 & self._kept_idx 생성"""
        rewards     = np.asarray(self.meta["rewards"],     dtype=float)
        clip_scores = np.asarray(self.meta["clip_scores"], dtype=float)
        prompts     = np.asarray(self.meta["prompts"])

        # 1) thresholds -------------------------------------------------------
        th = {}
        if "rewards" in self.filtering_criteria:
            q = self.filtering_criteria["rewards"]
            th["rewards_glob"] = np.quantile(rewards, q)
        if "clip_scores" in self.filtering_criteria:
            q = self.filtering_criteria["clip_scores"]
            th["clip_glob"] = np.quantile(clip_scores, q)

        if self.per_prompt_filtering_flag:
            th_per = defaultdict(dict)
            for p in np.unique(prompts):
                mask = prompts == p
                if "rewards" in self.filtering_criteria:
                    th_per[p]["rewards"] = np.quantile(rewards[mask], self.filtering_criteria["rewards"])
                if "clip_scores" in self.filtering_criteria:
                    th_per[p]["clip_scores"] = np.quantile(clip_scores[mask], self.filtering_criteria["clip_scores"])
            self.th_global, self.th_per_prompt = None, th_per
        else:
            self.th_global, self.th_per_prompt = {k.split("_")[0]:v for k,v in th.items()}, None

        # 2) filter -----------------------------------------------------------
        all_kept = [i for i in range(len(prompts)) if self._pass_filter(i)]
        kept_on  = [i for i in all_kept if self._epoch_of_idx(i) == self.current_epoch]
        kept_off = [i for i in all_kept if self._epoch_of_idx(i) <  self.current_epoch]

        if self.off_policy_subset_size and len(kept_off) > self.off_policy_subset_size:
            kept_off = random.sample(kept_off, self.off_policy_subset_size)
        if self.off_policy_subset_size == 0:
            kept_off.clear()
        
        
        self._kept_idx = kept_on + kept_off

        if verbose:
            self.logger.info(f"SearchDataset | kept {len(self._kept_idx):,} / {len(self.meta['prompts']):,}")
            
    def _epoch_of_idx(self, meta_idx: int) -> int:
        epoch, _, _ = self._index_map[meta_idx]
        return epoch

    # ――― filtering helper
    def _pass_filter(self, meta_idx: int) -> bool:
        p   = self.meta["prompts"][meta_idx]
        rew = float(self.meta["rewards"][meta_idx])
        clip= float(self.meta["clip_scores"][meta_idx])

        if self.th_per_prompt is not None:
            th = self.th_per_prompt[p]
        else:
            th = self.th_global

        if "rewards" in th and rew < th["rewards"]:
            return False
        if "clip_scores" in th and clip < th["clip_scores"]:
            return False
        return True

    # ――― fetch meta as dict
    def _meta_at(self, idx:int)->Dict:
        return {k: self.meta[k][idx] for k in self.meta}

    # ――― 다양도 선택 (메타만 사용, anchors==candidates 인 경우가 대부분)
    def _select_diverse_candidates_meta(self, anchors_meta, cand_meta, J:int):
        if J==0: return list(range(len(cand_meta)))
        A_emb = torch.stack([m["image_embedding"] for m in anchors_meta])  # [A,D]
        C_emb = torch.stack([m["image_embedding"] for m in cand_meta])     # [C,D]

        A_emb = F.normalize(A_emb.flatten(1), dim=1)
        C_emb = F.normalize(C_emb.flatten(1), dim=1)

        if len(anchors_meta):
            init_sim = C_emb @ A_emb.T
            min_dist = 1 - init_sim.max(dim=1).values
        else:
            min_dist = torch.ones(len(cand_meta))

        picked = []
        for _ in range(min(J, len(cand_meta))):
            pick = torch.argmax(min_dist).item()
            picked.append(pick)
            dist_new = 1 - (C_emb @ C_emb[pick:pick+1].T).squeeze(1)
            min_dist = torch.minimum(min_dist, dist_new)
            min_dist[pick] = -float("inf")
        return picked

    # ───────────────────────────── Dataset API ───────────────────────────────
    def __len__(self):
        return len(self._kept_idx)

    def __getitem__(self, idx:int):
        """idx → 실제 샘플(dict).  디스크에서 읽고 메모리 캐시에 올린다."""
        real_idx = self._kept_idx[idx]
        epoch, rank, local_idx = self._index_map[real_idx]

        # 1) 파일 cache 로드
        key = (epoch, rank)
        if key not in self._file_data_cache:
            file_path = self.buffer_root / f"epoch_{epoch}_rank_{rank}.pt"
            self._file_data_cache[key] = torch.load(file_path, map_location="cpu")["dataset"]
        sample = self._file_data_cache[key][local_idx]

        return sample
    
    def _calculate_prompts_stats(self):
        self.prompts_stats = defaultdict(lambda: {"rewards": [], "clip": [], "n": 0})

        for idx in self._kept_idx:
            p = self.meta["prompts"][idx]
            r = float(self.meta["rewards"][idx])
            c = float(self.meta["clip_scores"][idx])

            st = self.prompts_stats[p]
            st["rewards"].append(r)
            st["clip"].append(c)
            st["n"] += 1


    def report_dataset_stats(self):
        self.logger.info("")
        self.logger.info("────────────────── Dataset summary ────────────────────")
        total_kept = len(self._kept_idx)

        for p, st in self.prompts_stats.items():
            mean_r = np.nanmean(st["rewards"])
            mean_c = np.nanmean(st["clip"])
            n      = st["n"]

            # prompt 별 threshold 문자열
            th = self.th_per_prompt[p] if self.th_per_prompt else self.th_global
            th_str = ", ".join(f"{k} ≥ {v:.4f}" for k, v in th.items())

            self.logger.info(
                f"{p[:36]:<36} | R={mean_r:5.3f} | clip={mean_c:5.3f} "
                f"| n={n:4d} | threshold: {th_str}"
            )

        self.logger.info("")
        self.logger.info(f"Filtering criteria (quantile) : {self.filtering_criteria}")
        self.logger.info(f"Final sample count            : {total_kept}")
        self.logger.info("───────────────────────────────────────────────────────\n")

def _cleanup_old_buffers(buffer_dir: str, current_epoch: int, world_size: int, logger):
    """
    on-policy 모드(off_policy_subset_size == 0)일 때,
    이전 epoch(< current_epoch)의 데이터·메타 파일을 모두 삭제.
    """
    deleted = 0
    for e in range(current_epoch):            # 0 .. current_epoch-1
        for r in range(world_size):
            for stem in (
                f"epoch_{e}_rank_{r}.pt",         # data
                f"meta_epoch_{e}_rank_{r}.pt",    # meta  ← NEW
            ):
                fpath = os.path.join(buffer_dir, stem)
                try:
                    os.remove(fpath)
                    deleted += 1
                except FileNotFoundError:
                    continue

    if deleted:
        logger.info(f"[buffer-gc] removed {deleted} old buffer files")


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