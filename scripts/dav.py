import contextlib
import datetime
import gc
import itertools
import os
import tempfile
import time
import warnings
from collections import defaultdict
from concurrent import futures
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import wandb
from absl import app, flags
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from ml_collections import config_flags
from PIL import Image

import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import search_pipeline_with_logprob
from utils import generate_evaluation_samples

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/dav.py", "Training configuration.")

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
        f'_KL={config.train.train_kl}'
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

    accumulation_steps = num_train_timesteps

    
    plugin = GradientAccumulationPlugin(num_steps=accumulation_steps * config.train.accumulation_multipler, sync_with_dataloader=False)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
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

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    

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
    logger.info(f"  Frequency for save = {config.save_freq}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Number of MLE training steps per epoch = {config.train.improve_steps}")
    logger.info(f"  Kullback-Liebler divergence coefficient = {config.train.train_kl}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    
    global_step = 0
    
    for epoch in range(first_epoch, config.num_epochs + 1):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompts = []
        images_list = []
        kl_div_list = []
        num_images_per_prompt = config.sample.num_batches_per_epoch // config.sample.num_prompts_per_batch
        
        prompts_history = []
        prompts_metadata_history = []
        prior_history = []
        
        idxs = np.random.choice(num_prompts, size=config.sample.num_prompts_per_batch, replace=False).tolist()
        prompts = [prompts_total[i] for i in idxs]  

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
                    shape = (config.search.duplicate, 4, 64, 64)
                else:
                    shape = (1, 4, 64, 64)
                init_latents = torch.randn(shape, device=accelerator.device)
                pipeline.batch_size = 1
                images, _, latents, log_probs, kl_divs, prior, _ = search_pipeline_with_logprob(
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
                kl_div_list.append(kl_divs)
                prompts_metadata_history.append(prompt_metadata)

            latents = torch.stack(
                latents, dim=1
            )
            log_probs = torch.stack(log_probs, dim=1)
            traj_kl_divs = torch.stack(kl_div_list[-1]).sum().view(1, -1)
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.batch_size, 1
            )

            rewards = executor.submit(reward_fn, images, prompt, prompt_metadata)

            time.sleep(0)

            samples.append(
                {
                    "prompts": prompt,
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
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
            eval_results = {key: torch.as_tensor(value, device=accelerator.device) for key, value in sample.items() if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards", "prompts"]}
            sample.update(eval_results)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        prompts_list = [sample["prompts"] for sample in samples]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()if k not in ['tree', 'prompts']}
        samples['prompts'] = prompts_list
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
                f"_{prompt[:40].replace(os.sep,'_')}"
                f"_{samples['rewards'][i]:.4f}.jpg"
            )
            pil_img = Image.fromarray((image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil_img.save(os.path.join(search_dir, filename))
        if dist.is_initialized():
            accelerator.wait_for_everyone()

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
        traj_kl_divs = accelerator.gather(samples["traj_kl_divs"]).cpu().numpy()
        elbo = rewards - traj_kl_divs

        log_dict = {
            "reward": rewards,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "traj_kl_divs": traj_kl_divs,
            "traj_kl_divs_mean": traj_kl_divs.mean(),
            "traj_kl_divs_std": traj_kl_divs.std(),
            "elbo": elbo,
            "elbo_mean": elbo.mean(),
            "elbo_std": elbo.std(),
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
            accelerator.log(log_dict, step=global_step)

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config.sample.batch_size * config.sample.num_batches_per_epoch
        )
        assert num_timesteps == config.sample.num_steps

        samples_batched = {
            k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) if k not in ['prompts'] else v
            for k, v in samples.items()
        }

        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        del samples
        gc.collect()
        torch.cuda.empty_cache()    
        time.sleep(1)
    
        #################### TRAINING ####################
        
        for improve_steps in range(config.train.improve_steps):
            pipeline.unet.train()
            info = defaultdict(list)
            
            for sample in tqdm(
                samples_batched,
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
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                timesteps.to(torch.int64),
                                latents,
                                eta=config.sample.eta,
                                prev_sample=next_latents
                            )

                        loss = -log_prob
                        info["mle_loss"].append(loss.detach())

                        if config.train.train_kl > 0:
                            with autocast():
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
                            kl_loss = config.train.train_kl * F.mse_loss(noise_pred, ref_noise_pred.detach())
                            loss = loss + kl_loss
                            info["kl_loss"].append(kl_loss.detach())

                        accelerator.backward(loss)
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
                    num_images_per_prompt=4
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
                    eval_images = eval_images_list[0]
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
                del eval_samples
                del eval_images_list
                del eval_rewards
                del eval_images_tensor

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            gc.collect()
            torch.cuda.empty_cache()

        del samples_batched
        gc.collect()
        torch.cuda.empty_cache()
        if epoch != 0 and epoch % config.save_freq == 0:
            accelerator.save_state()



if __name__ == "__main__":
    app.run(main)
