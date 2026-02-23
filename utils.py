import os
import time
import numpy as np
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch
from functools import partial
import tqdm

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


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
    Generate evaluation images, compute log_prob and rewards; return eval_samples
    and eval_images_list.
    """
    eval_images_list = []
    eval_samples = []
    eval_rewards_list = []

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

        # compute reward asynchronously
        eval_rewards_future = executor.submit(reward_fn, eval_images, prompts_history[i % len(prompts_history)], prompts_metadata_history[i % len(prompts_history)])
        time.sleep(0)  # allow async call to start
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size, 1
        )
        eval_samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": eval_latents[:, :-1],   # latent before each timestep
                "next_latents": eval_latents[:, 1:],  # latent after each timestep
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