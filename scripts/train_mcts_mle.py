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

from buffer import PrioritizedReplayBuffer

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/svdd_aesthetic_mle.py", "Training configuration.")

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

    if config.multistep_mdp:
        if config.train.type == 'energy_based_negative_gradient':
            accumulation_steps = num_train_timesteps * 2
        else:
            accumulation_steps = num_train_timesteps
    else:
        accumulation_steps = int(config.train.total_batch_size / (torch.cuda.device_count()))
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps= accumulation_steps # int(config.train.total_batch_size / (torch.cuda.device_count()))
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
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
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
            

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # evals = executor.submit(eval_fn, images, prompts)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
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
                    "tree": tree
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
            eval_results = {key: torch.as_tensor(value, device=accelerator.device) for key, value in sample.items() if key not in ["prompt_ids", "prompt_embeds", "timesteps", "latents", "next_latents", "log_probs", "rewards", "tree"]}
            sample.update(eval_results)
            # del sample["evals"]

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        trees = [sample['tree'] for sample in samples]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()if k != 'tree'}
        samples['trees'] = trees

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs

        save_dir = f'images/{config.run_name}'
        os.makedirs(save_dir, exist_ok=True) 

        for i, (image, prompt) in enumerate(zip(images_list, prompts_history)):
            pil = Image.fromarray(
                (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil.save(os.path.join(save_dir, f"G:{epoch+1}_{prompt}_search_{(i + 1) * (accelerator.local_process_index + 1)}_{samples['rewards'][i]:.4f}.jpg"))


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
        # metrics = {key: accelerator.gather(torch.stack([s[key] for s in samples])).cpu().numpy() for key in eval_results.keys()}
        
        log_dict = {
            "reward": rewards,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
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
        
        samples_batched = {
            k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) if k != 'trees' else v
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
                                    
                                loss = -search_log_prob
                                info["positive_loss"].append(loss)
                                accelerator.backward(loss)
                                optimizer.step()
                                optimizer.zero_grad()
                                
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
                                info["negative_loss"].append(loss)                                
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
            
            for i, (image, prompt) in enumerate(zip(eval_images_tensor, prompts_total)):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil.save(os.path.join(save_dir, f"G:{epoch+1}_I:{improve_steps+1}_{prompt}_{(i + 1) * (accelerator.local_process_index + 1)}_eval_{eval_rewards[i]:.4f}.jpg"))
                
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
