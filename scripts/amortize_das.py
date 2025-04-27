from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import random
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
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
import torchvision
from typing import Optional, Callable
from torchvision import transforms

from buffer import PrioritizedReplayBuffer

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/das_amortization.py", "Training configuration.")

logger = get_logger(__name__)


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
        eval_log_probs = torch.stack(eval_log_probs)

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
    
    # 비동기로 계산된 reward를 기다리고, 결과를 텐서로 변환
    for sample in tqdm(
        eval_samples,
        desc="Waiting for rewards",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        eval_rewards, _ = sample["eval_rewards"].result()
        sample["eval_rewards"] = torch.as_tensor(eval_rewards, device=accelerator.device)
        eval_rewards_list.append(sample["eval_rewards"])
        # 추가 eval 결과를 텐서로 변환 (여기서는 key가 지정된 항목들을 제외한 나머지)
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


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, neg_image, reward, neg_reward, prompt, prompt_ids = self.samples[idx]
        image = self.transform(image)
        neg_image = self.transform(neg_image)
        
        return {
            "win_image": image,
            "lose_image": neg_image,
            "reward": reward,
            "neg_reward": neg_reward,
            "prompt": prompt,
            "prompt_ids": prompt_ids
        }

    
    
    
    
def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    config.run_name = (
        f'{config.reward_fn}'
        f'_{config.train.type}'
        f'_B={config.sample.batch_size * config.sample.num_batches_per_epoch * torch.cuda.device_count()}'
        f'_M={config.search.duplicate}'
        f'_I={config.train.gradient_steps_per_improve_step}'
        f'_KL={config.train.kl_coef}'
        f'_G={config.search.value_gradient}:{config.search.kl_lagrangian_coef}'
        f'_{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}'
        f'_{config.run_name}'
    )
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    

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
    accumulation_steps = int(config.train.total_batch_size / (torch.cuda.device_count())) if not config.train.sft_negative_gradient else int(config.train.total_batch_size / (torch.cuda.device_count()))
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=accumulation_steps
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="das-mle",
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
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
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

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
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
    
    
    img_folder = '/home/jaewoo/research/diffusion-mcts/DAS/logs/DAS_SD/aesthetic/2025.04.21_20.48.19'
    image_names = [file for file in os.listdir(img_folder + "/eval_vis") if (file.endswith(('png', 'jpg', 'jpeg')) and not "ess" in file and not "intermediate_rewards" in file)]
    negative_image_names = [file for file in os.listdir(img_folder + "/eval_neg") if (file.endswith(('png', 'jpg', 'jpeg')) and not "ess" in file and not "intermediate_rewards" in file)]
    
    prompt_to_samples = defaultdict(list)
    
    for image_name, neg_image_name in zip(image_names, negative_image_names):
        image_path = os.path.join(img_folder, "eval_vis", image_name)
        neg_image_path = os.path.join(img_folder, "eval_neg", neg_image_name)

        image = Image.open(image_path).convert("RGB")
        neg_image = Image.open(neg_image_path).convert("RGB")

        prompt = image_name.split("|")[0].split("_")[-1][:-1]
        reward = float(image_name.split(":")[-1][:-4])
        neg_reward = float(neg_image_name.split(":")[-1][:-4])
        prompt_ids = pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids[0]
        prompt_to_samples[prompt].append((image, neg_image, reward, neg_reward, prompt, prompt_ids))

    filtered_samples = []
    for prompt, samples in prompt_to_samples.items():
        rewards = [s[2] for s in samples]  # s[2] = reward
        threshold = np.percentile(rewards, config.data.reward_filtering_percentile)
        for sample in samples:
            if sample[2] >= threshold:
                filtered_samples.append(sample)
        
    print(f"{'Prompt'.ljust(20)}{'Count':<8}{'Mean':<10}{'Std':<10}{'Min':<10}{'Median':<10}{'Max':<10}")
    for prompt, samples in prompt_to_samples.items():
        rewards = [s[2] for s in samples]
        count = len(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards)
        rmin = np.min(rewards)
        median = np.median(rewards)
        rmax = np.max(rewards)
        
        print(f"{prompt.ljust(20)}{str(count):<8}{mean:.4f}    {std:.4f}    {rmin:.4f}    {median:.4f}    {rmax:.4f}")

    train_transforms = transforms.Compose(
            [
                transforms.Resize(config.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(config.data.resolution) if config.data.random_crop else transforms.CenterCrop(config.data.resolution),
                transforms.Lambda(lambda x: x) if config.data.no_hflip else transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )
    
    train_dataset = ImageDataset(filtered_samples, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )
    

    if config.train.kl_coef > 0 or config.train.type == 'dpo':
        unet_pretrained = UNet2DConditionModel.from_pretrained(
            config.pretrained.model,
            revision=config.pretrained.revision,
            subfolder="unet",
        ).to(accelerator.device, dtype=inference_dtype)

        # Prepare everything with our `accelerator`.
        unet, optimizer, unet_pretrained, train_dataloader = accelerator.prepare(unet, optimizer, unet_pretrained, train_dataloader)
        
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
        

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
    logger.info(f"  Kullback-Liebler divergence coefficient = {config.train.kl_coef}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert config.train.total_batch_size % accelerator.num_processes == 0
    # assert config.eval.num_images_per_prompt % accelerator.num_processes == 0, "eval.num_prompts_per_batch must be devided  for now"


    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    
    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### TRAINING ####################

        pipeline.unet.train()
        mse_loss = torch.nn.MSELoss()
        info = defaultdict(list)
        
        for samples in tqdm(
            train_dataloader,
            desc=f"EPOCH {epoch + 1} ",
            position=0,
            leave=True,
            disable=not accelerator.is_local_main_process,
        ):
            with accelerator.accumulate(unet):
                with autocast():
                    prompt_embeds = pipeline.text_encoder(samples["prompt_ids"])[0]
                    positive_latents = pipeline.vae.encode(samples["win_image"].to(inference_dtype)).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    negative_latents = pipeline.vae.encode(samples["lose_image"].to(inference_dtype)).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    if config.train.type == 'sft':
                        if config.train.cfg:
                            # concat negative prompts to sample prompts to avoid two forward passes
                            embeds = torch.cat(
                                [train_neg_prompt_embeds, prompt_embeds]
                            )
                        else:
                            embeds = prompt_embeds

                        if config.train.cfg:
                            clean_latents = positive_latents
                            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (config.train.batch_size,), device=accelerator.device)
                            noise = torch.randn_like(clean_latents, dtype=torch.float32)
                            noised_latents = pipeline.scheduler.add_noise(clean_latents, noise, timesteps)
                            noise_pred = unet(torch.cat([noised_latents] * 2), torch.cat([timesteps] * 2), embeds).sample
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

                            loss = mse_loss(noise_pred, noise)

                        else:
                            raise NotImplementedError("Not implemented yet")
                        

                        if config.train.kl_coef > 0:
                            kl_loss = config.train.kl_coef * mse_loss(noise_pred, ref_noise_pred.detach())
                            loss = loss + kl_loss
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                    elif config.train.type == 'dpo':
                        negative_latents = pipeline.vae.encode(sample["lose_image"].to(inference_dtype)).latent_dist.sample() * pipeline.vae.config.scaling_factor
                        clean_latents = torch.cat([positive_latents, negative_latents])
                        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (config.train.batch_size,), device=accelerator.device)
                        timesteps = timesteps.long().chunk(2)[0].repeat(2)
                        noise = torch.randn_like(clean_latents)
                        noised_latents = pipeline.scheduler.add_noise(clean_latents, noise, timesteps)
                        embeds = samples["prompt_embeds"].repeat(2, 1, 1)

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
            if not config.train.sft_negative_gradient:
                if accelerator.sync_gradients:
                    # assert j == (config.train.total_batch_size // accelerator.num_processes) - 1
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch})
                    accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
                    accelerator.clip_grad_norm_(
                        unet.parameters(), config.train.max_grad_norm
                    )
            optimizer.step()
            optimizer.zero_grad()
            
            with accelerator.accumulate(unet):
                with autocast():
                    if config.train.type == 'sft':
                        if config.train.cfg:
                            if config.train.sft_negative_gradient:
                                clean_negative_latents = negative_latents
                                noise = torch.randn_like(clean_negative_latents, dtype=torch.float32)
                                noised_negative_latents = pipeline.scheduler.add_noise(clean_negative_latents, noise, timesteps)
                                noise_negative_pred = unet(torch.cat([noised_negative_latents] * 2), torch.cat([timesteps] * 2), embeds).sample
                                noise_negative_pred_uncond, noise_negative_pred_text = noise_negative_pred.chunk(2)
                                noise_negative_pred = (
                                    noise_negative_pred_uncond
                                    + config.sample.guidance_scale
                                    * (noise_negative_pred_text - noise_negative_pred_uncond)
                                )
                                loss = -mse_loss(noise_negative_pred, noise)
                                accelerator.backward(loss)
                                if accelerator.sync_gradients:
                                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                    info = accelerator.reduce(info, reduction="mean")
                                    info.update({"epoch": epoch})
                                    accelerator.log(info, step=global_step)
                                    global_step += 1
                                    info = defaultdict(list)
                                    accelerator.clip_grad_norm_(
                                        unet.parameters(), config.train.max_grad_norm
                                    )
                                optimizer.step()
                                optimizer.zero_grad()
                        else:
                            raise NotImplementedError("Not implemented yet")
            

            # if accelerator.sync_gradients:
                # assert (int(config.train.gradient_steps_per_improve_step / (accelerator.num_processes * config.train.batch_size))) % config.train.gradient_accumulation_steps == 0
        
        if epoch % 10 == 0:
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
            
            save_dir = f'images/{config.run_name}'
            os.makedirs(save_dir, exist_ok=True) 
            
            for i, (image, prompt) in enumerate(zip(eval_images_tensor, prompts_total)):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(save_dir, f"G:{epoch+1}_{prompt}_{(i + 1) * (accelerator.local_process_index + 1)}_eval_{eval_rewards[i]:.4f}.jpg"))
                
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
                                zip(prompts_total[:len(eval_images_list[0])], eval_rewards[:len(eval_images_list[0])])
                            )
                        ],
                    },
                    step=global_step,
                )
            # log rewards and images
            log_dict = {
                "eval_reward": eval_rewards,
                "eval_reward_mean": eval_rewards.mean(),
                "eval_reward_std": eval_rewards.std(),
            }
            accelerator.log(log_dict, step=global_step)
            # make sure we did an optimization step at the end of the inner epoch
        assert accelerator.sync_gradients
        if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()




if __name__ == "__main__":
    app.run(main)
