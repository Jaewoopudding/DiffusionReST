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
config_flags.DEFINE_config_file("config", "config/svdd_aesthetic_mle.py", "Training configuration.")

logger = get_logger(__name__)


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
        f'_I={config.train.improve_steps}'
        f'_{datetime.datetime.now().strftime("%Y.%m.%d")}'
        f'_{config.run_name}'
        f'_S={config.seed}'
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
    plugin = GradientAccumulationPlugin(num_steps=accumulation_steps * config.train.accumulation_multipler, sync_with_dataloader=False)
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
            project_name=config.reward_fn,
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
    # assert config.train.total_batch_size % accelerator.num_processes == 0
    # assert config.eval.num_images_per_prompt % accelerator.num_processes == 0, "eval.num_prompts_per_batch must be devided  for now"


    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    
    global_step = 0
    
    for epoch in range(first_epoch, config.num_epochs + 1):
        #################### SAMPLING ####################
        on_policy_dataset_per_gpu = []
        buffers = [PrioritizedReplayBuffer(capacity=10000, priority="rewards") for _ in range(num_prompts)]
        pipeline.unet.eval()
        samples = []
        eval_samples = []
        prompts = []
        images_list = []
        advantages_list = []
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
                if config.initial_search:
                    shape = (config.search.duplicate * config.search.nfe_per_action, 4, 64, 64)
                else:
                    shape = (config.search.nfe_per_action, 4, 64, 64)
                init_latents = torch.randn(shape, device=accelerator.device)
                pipeline.batch_size = 1
                images, _, latents, log_probs, advantages, next_weighted_mean_states, mean_advantages, prior, _ = tree_pipeline_with_logprob(
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
                prompts_metadata_history.append(prompt_metadata)
                next_weighted_mean_states_list.append(next_weighted_mean_states)

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 4, 64, 64)
            next_weighted_mean_states = torch.stack(next_weighted_mean_states, dim=1)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            mean_advantages = torch.stack(mean_advantages_list[0]).view(1, -1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )  # (batch_size, num_steps)
            if config.train.kl_lagrangian_coef:
                advantages = advantages.unsqueeze(0) / config.train.kl_lagrangian_coef
            else:
                advantages = advantages.unsqueeze(0) * 0

            rewards = executor.submit(reward_fn, images, prompt, prompt_metadata)

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
                    "next_weighted_mean_states": next_weighted_mean_states[:, :-1],
                    "mean_advantages": mean_advantages[:, :-1],
                    "log_probs": log_probs,
                    "rewards": rewards,
                    # "evals": evals,
                    "final_embedding": latents[:, -1],
                    "image_embedding": embedded_images,
                    "advantages": advantages,
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
            eval_results = {key: torch.as_tensor(value, device=accelerator.device) for key, value in sample.items() if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards", "advantages", "prompts"]}
            sample.update(eval_results)
            # del sample["evals"]


        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        prompts_list = [sample["prompts"] for sample in samples]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()if k not in ['tree', 'prompts']}
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
            # log rewards and images
            log_dict = {
                "eval_reward": eval_rewards,
                "eval_reward_mean": eval_rewards.mean(),
                "eval_reward_std": eval_rewards.std(),
            }

            if config.reward_fn == "aesthetic_score_diff_clipped":
                log_dict = {
                    "eval_reward": eval_rewards + 8.5,
                    "eval_reward_mean": eval_rewards.mean() + 8.5,
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
        # samples["advantages"] = (
        #     torch.as_tensor(advantages)
        #     .reshape(accelerator.num_processes, -1)[accelerator.process_index]
        #     .to(accelerator.device)
        # )

        # shuffle along time dimension independently for each sample
        total_batch_size, num_timesteps = samples["timesteps"].shape
        # if config.multistep_mdp:
        #     perms = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])
        #     
        #     for key in ["timesteps", "latents", "next_latents", "log_probs", "eval_latents", "eval_next_latents", "eval_log_probs", "advantages", "mean_advantages", "next_weighted_mean_states"]:
        #         samples[key] = samples[key][
        #             torch.arange(total_batch_size, device=accelerator.device)[:, None],
        #             perms,
        #         ]


        samples_batched = {
            k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) if k not in ['prompts'] else v
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


        # SAVE ON_POLICY DATASET
        samples_batched_cpu = [_to_cpu(s) for s in samples_batched]
        for i, sample in enumerate(samples_batched_cpu):
            on_policy_dataset_per_gpu.append(sample)

        on_policy_dataset_per_gpu  = _strip_unpicklable(on_policy_dataset_per_gpu)
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        buffer_dir = os.path.join("buffer", config.run_name)
        os.makedirs(buffer_dir, exist_ok=True)
        buffer_path = os.path.join(buffer_dir, f"epoch_{epoch}_rank_{rank}.pt")
        meta_path = os.path.join(buffer_dir, f"meta_epoch_{epoch}_rank_{rank}.pt")

        torch.save({
            "dataset": on_policy_dataset_per_gpu,
        }, buffer_path)
        
        meta = {
            "prompts"        : [s["prompts"] for s in samples_batched_cpu],       # str list (길이 = B)
            "rewards"        : torch.stack([s["rewards"].cpu()        for s in samples_batched_cpu]),
            "clip_scores"    : torch.stack([s["clip_scores"].cpu()    for s in samples_batched_cpu]),
            "image_embedding": torch.stack([s["image_embedding"].cpu() for s in samples_batched_cpu]),
        }
        torch.save(meta, meta_path)     #  buffer/meta_epoch_{E}_rank_{R}.pt  로 저장
            
        if dist.is_initialized():
            accelerator.wait_for_everyone()    # 모든 rank 저장 끝날 때까지 대기

        if accelerator.is_main_process and config.buffer.off_policy_subset_size == 0 and epoch > 0:
            _cleanup_old_buffers(buffer_dir, epoch, world_size, logger)  # ← NEW!

        if dist.is_initialized():
            accelerator.wait_for_everyone()    # 삭제 끝나면 다시 동기화

        gathered_meta = {k: [] for k in meta}    # {prompts:[], rewards:[], …}

        for e in range(epoch + 1):                      # 0 ~ current epoch
            for r in range(world_size):
                meta_path = os.path.join(buffer_dir, f"meta_epoch_{e}_rank_{r}.pt")
                data_path = os.path.join(buffer_dir, f"epoch_{e}_rank_{r}.pt")

                # ① 메타가 없으면 당연히 skip
                if not os.path.exists(meta_path):
                    continue
                # ② 메타는 있는데 데이터가 없으면 skip  (※ NEW)
                if not os.path.exists(data_path):
                    logger.warning(f"[buffer] meta만 있고 data가 없어 건너뜀 → {meta_path}")
                    continue

                m = torch.load(meta_path, map_location="cpu")
                for k in gathered_meta:
                    gathered_meta[k].append(m[k])

        if dist.is_initialized():
            accelerator.wait_for_everyone()

        gathered_meta["prompts"]         = list(itertools.chain.from_iterable(gathered_meta["prompts"]))
        gathered_meta["rewards"]         = torch.cat(gathered_meta["rewards"])
        gathered_meta["clip_scores"]     = torch.cat(gathered_meta["clip_scores"])
        gathered_meta["image_embedding"] = torch.cat(gathered_meta["image_embedding"])

        replaybuffer = SearchDataset(
            metadata   = gathered_meta,
            buffer_path  = os.path.join("buffer", config.run_name),
            epoch     = epoch,
            per_prompt_filtering_flag = config.buffer.per_prompt_filtering_flag,
            per_prompt_select_flag = config.buffer.per_prompt_select_flag,
            filtering_criteria  = {"rewards": config.buffer.reward_filtering_criteria, "clip_scores": config.buffer.clip_score_filtering_criteria},
            off_policy_subset_size     = 0 if epoch == 0 else config.buffer.off_policy_subset_size, # off policy sample 중 prompt당 3개씩 가져오기. 
            batch_size_subset   = 2048,
            logger              = logger
        )

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
        del meta
        del samples_batched_cpu
        del samples_batched
        del samples
        
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
                    raise NotImplementedError
                                    
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
                                    latents = sample['latents'][:, j]
                                    timesteps = sample['timesteps'][:, j]
                                    next_latents = sample['next_latents'][:, j]
                                    advantages = sample['advantages'][:, j]
                                    
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
                                loss = -torch.exp(advantages).clamp(max=5.) * search_log_prob
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

                elif config.train.type == 'boltzmann_loss_fn': 
                    raise NotImplementedError

            if (improve_steps == config.train.improve_steps - 1) and ((epoch + 1) % config.eval.eval_freq == 0):
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
                    num_images_per_prompt=1 if (epoch != config.num_epochs - 1) and (epoch != 100) else 4
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
                
                if config.reward_fn == "aesthetic_score_diff_clipped":
                    log_dict = {
                        "eval_reward": eval_rewards + 8.5,
                        "eval_reward_mean": eval_rewards.mean() + 8.5,
                        "eval_reward_std": eval_rewards.std(),
                    }
                
                accelerator.log(log_dict, step=global_step)
                del eval_samples
                del eval_images_list
                del eval_rewards
                del eval_images_tensor

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            accelerator._dataloaders.clear()
            gc.collect()
            torch.cuda.empty_cache()

        del buffers
        del gathered_meta
        del replaybuffer
        del dataloader    
        gc.collect()
        torch.cuda.empty_cache()
        if epoch != 0 and epoch % config.save_freq == 0:
            accelerator.save_state()



if __name__ == "__main__":
    app.run(main)
