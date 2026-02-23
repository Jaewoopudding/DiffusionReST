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

# Import from utils.py
from utils import collate_without_tree_prompt, SearchDataset, _cleanup_old_buffers, generate_evaluation_samples

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/sample.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    if os.path.exists(config.checkpoint_dir):
        checkpoints = []
        for item in os.listdir(config.checkpoint_dir):
            if item.startswith("checkpoint_"):
                checkpoint_path = os.path.join(config.checkpoint_dir, item)
                if os.path.isdir(checkpoint_path):
                    checkpoints.append(checkpoint_path)
        checkpoints.sort(key=lambda x: int(x.split("_")[-1]))  # checkpoint 번호로 정렬
        
        # 체크포인트 디렉토리의 상위 디렉토리 이름을 run_name으로 사용
        checkpoint_parent_dir = os.path.basename(os.path.dirname(config.checkpoint_dir))
        config.run_name = checkpoint_parent_dir + "_" + datetime.datetime.now().strftime("%Y.%m.%d.%H.%M")
    else:
        # 체크포인트 디렉토리가 없을 경우 기본 run_name 사용
        config.run_name = (
            f'{config.reward_fn}'
            f'_B={config.sample.batch_size * config.sample.num_batches_per_epoch * torch.cuda.device_count()}'
            f'_M={config.search.duplicate}'
            f'_KL={config.train.kl_coef}'
            f'_G={config.search.value_gradient}:{config.search.kl_lagrangian_coef}'
            f'_I={config.train.improve_steps}'
            f'_{datetime.datetime.now().strftime("%Y.%m.%d")}'
            f'_{config.run_name}'
            f'_S={config.seed}'
            f'_T={datetime.datetime.now().year}_{datetime.datetime.now().month:02d}_{datetime.datetime.now().day:02d}_{datetime.datetime.now().hour:02d}_{datetime.datetime.now().minute:02d}'
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
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        # gradient_accumulation_steps=accumulation_steps, # int(config.train.total_batch_size / (torch.cuda.device_count()))
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=f"{config.reward_fn}_eval",
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

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    def save_model_hook(models, weights, output_dir):
        try:
            pipeline.unet.save_attn_procs(output_dir)
            weights.clear()
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
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.clear()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
    

    unet_pretrained = UNet2DConditionModel.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        subfolder="unet",
    ).to(accelerator.device, dtype=inference_dtype)
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, unet_pretrained = accelerator.prepare(unet, optimizer, unet_pretrained)
        

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

    # 체크포인트 디렉토리에서 모든 체크포인트 찾기

    global_step = 0
    
    # k번째마다 체크포인트 샘플링 (k=2이면 1,3,5,... / k=3이면 2,5,8,...)
    checkpoint_interval = getattr(config, 'checkpoint_interval', 1)  # 기본값은 1 (모든 체크포인트)
    
    # checkpoint_interval에 따라 필터링
    if checkpoint_interval > 1:
        # checkpoint_interval=2이면 인덱스 0,2,4,... 선택 (체크포인트 1,3,5,...)
        # checkpoint_interval=3이면 인덱스 1,4,7,... 선택 (체크포인트 2,5,8,...)
        start_idx = checkpoint_interval - 1
        selected_checkpoints = checkpoints[start_idx::checkpoint_interval]
        logger.info(f"Evaluating every {checkpoint_interval}-th checkpoint starting from checkpoint {start_idx+1}")
        logger.info(f"Selected checkpoints: {[int(cp.split('_')[-1]) for cp in selected_checkpoints]}")
    else:
        selected_checkpoints = checkpoints
        logger.info("Evaluating all checkpoints")
    
    # 각 선택된 체크포인트에서 샘플링
    for checkpoint_path in selected_checkpoints:
        checkpoint_num = int(checkpoint_path.split("_")[-1])
        logger.info(f"Loading checkpoint {checkpoint_num} from {checkpoint_path}")
        
        # 체크포인트 로드
        accelerator.load_state(checkpoint_path)
        
        # 이 체크포인트에서 샘플링을 위한 epoch 설정
        epoch = checkpoint_num
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        eval_samples = []
        prompts = []
        images_list = []
        kl_div_list = []
        next_weighted_mean_states_list = []
        eval_images_list = []
        num_images_per_prompt = config.sample.num_batches_per_epoch // config.sample.num_prompts_per_batch
        
        prompts_history = []
        prompts_metadata_history = []
        prior_history = []
        mean_advantages_list = []
        
        idxs = np.random.choice(num_prompts, size=config.sample.num_prompts_per_batch, replace=False).tolist()
        prompts = [prompts_total[i] for i in idxs]  

        image_embedder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        image_embedder.to(accelerator.device, dtype=inference_dtype)
        
        
        if config.search_or_zeroshot:
            prompt = prompts_total
            
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
                if config.initial_search:
                    shape = (config.search.duplicate * config.search.nfe_per_action, 4, 64, 64)
                else:
                    shape = (config.search.nfe_per_action, 4, 64, 64)
                init_latents = torch.randn(shape, device=accelerator.device)
                pipeline.batch_size = 1
                images, _, latents, log_probs, advantages, next_weighted_mean_states, mean_advantages, kl_divs, prior, _ = tree_pipeline_with_logprob(
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
                mean_advantages_list.append(mean_advantages)
                kl_div_list.append(kl_divs)
                prompts_metadata_history.append(prompt_metadata)
                next_weighted_mean_states_list.append(next_weighted_mean_states)
            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            next_weighted_mean_states = torch.stack(next_weighted_mean_states, dim=1)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            mean_advantages = torch.stack(mean_advantages_list[-1]).view(1, -1)
            traj_kl_divs = torch.stack(kl_div_list[-1]).sum().view(1, -1) # batchsize = n failure case
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)
            if config.train.kl_lagrangian_coef:
                advantages = advantages.unsqueeze(0) / config.train.kl_lagrangian_coef
            else:
                advantages = advantages.unsqueeze(0) * 0
            rewards = executor.submit(reward_fn, images, prompt, prompt_metadata)
            time.sleep(0)
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
                    "next_weighted_mean_states": next_weighted_mean_states[:, :-1],
                    "mean_advantages": mean_advantages[:, :-1],
                    "log_probs": log_probs,
                    "rewards": rewards,
                    # "evals": evals,
                    "final_embedding": latents[:, -1],
                    "advantages": advantages,
                    "traj_kl_divs": traj_kl_divs,
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
                eval_results = {key: torch.as_tensor(value, device=accelerator.device) for key, value in sample.items() if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards", "advantages", "prompts"]}
                sample.update(eval_results)
            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            prompts_list = [sample["prompts"] for sample in samples]
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()if k not in ['tree', 'prompts']}
            samples['prompts'] = prompts_list
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            del image_embedder
            gc.collect()
            torch.cuda.empty_cache()
            
            save_dir = f'eval_images/{config.run_name}'
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
            traj_kl_divs = accelerator.gather(samples["traj_kl_divs"]).cpu().numpy()
            elbo = rewards - traj_kl_divs
            # metrics = {key: accelerator.gather(torch.stack([s[key] for s in samples])).cpu().numpy() for key in eval_results.keys()}
            log_dict = {
                "reward": rewards,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "clip_score": clip_scores,
                "clip_score_mean": clip_scores.mean(),
                "clip_score_std": clip_scores.std(),
                "traj_kl_divs": traj_kl_divs,
                "traj_kl_divs_mean": traj_kl_divs.mean(),
                "traj_kl_divs_std": traj_kl_divs.std(),
                "elbo": elbo,
                "elbo_mean": elbo.mean(),
                "elbo_std": elbo.std(),
            }
            
            if config.reward_fn == "aesthetic_score_diff_clipped":
                log_dict = {
                    "reward": rewards + 8.5,
                    "reward_mean": rewards.mean() + 8.5,
                    "reward_std": rewards.std(),
                    "clip_score": clip_scores,
                    "clip_score_mean": clip_scores.mean(),
                    "clip_score_std": clip_scores.std(),
                }
            accelerator.log(log_dict, step=global_step)
        if not config.search_or_zeroshot:
            prompts = prompts_total
            # prior_history 생성 - 한 번에 랜덤하게 생성
            prior_history = torch.randn(
                (len(prompts), config.sample.batch_size, 4, 64, 64),
                device=accelerator.device,
                dtype=torch.float16
            )
            
            # prompts_metadata_history 생성 - prompts와 같은 길이의 빈 딕셔너리 리스트
            prompts_metadata_history = [{} for _ in range(len(prompts))]
            
            eval_samples, eval_images_list, eval_rewards = generate_evaluation_samples(
                pipeline=pipeline,
                sample_neg_prompt_embeds=sample_neg_prompt_embeds,
                config=config,
                accelerator=accelerator,
                epoch=epoch,
                reward_fn=reward_fn,
                executor=executor,
                prompts_history=prompts,
                prior_history=prior_history,
                prompts_metadata_history=prompts_metadata_history,
                autocast=autocast,
                num_images_per_prompt=config.sample.count
            )
            save_dir = f'eval_images/{config.run_name}/checkpoint_{checkpoint_num}'
            os.makedirs(save_dir, exist_ok=True) 
            # 각 체크포인트에서 이미지 저장
            for i, image in enumerate(eval_images_list):
                # 프롬프트를 파일명에 안전하게 포함시키기 위한 처리
                prompt = prompts[i % len(prompts)]  # 프롬프트 가져오기
                # 특수문자 제거 및 길이 제한 (최대 50자)
                safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_'))[:50]
                safe_prompt = safe_prompt.replace(' ', '_')  # 공백을 언더스코어로 변경
                
                pil = Image.fromarray(
                    (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(save_dir, f"checkpoint_{checkpoint_num}_eval_{i}_{safe_prompt}_{eval_rewards[i]:.4f}.jpg"))
            
            # log rewards and images
            log_dict = {
                f"eval_reward": eval_rewards,
                f"eval_reward_mean": eval_rewards.mean(),
                f"eval_reward_std": eval_rewards.std(),
            }
            accelerator.log(log_dict, step=global_step)
            global_step += 1
            
            logger.info(f"Checkpoint {checkpoint_num} sampling completed. Saved {len(eval_images_list)} images to {save_dir}")
        
        gc.collect()
        torch.cuda.empty_cache()    
        time.sleep(1)


if __name__ == "__main__":
    app.run(main)
