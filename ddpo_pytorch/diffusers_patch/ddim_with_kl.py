# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/schedulers/scheduling_ddim.py
# with the following modifications:
# - It computes and returns the log prob of `prev_sample` given the UNet prediction.
# - Instead of `variance_noise`, it takes `prev_sample` as an optional argument. If `prev_sample` is provided,
#   it uses it to compute the log prob.
# - Timesteps can be a batched torch.Tensor.

from typing import Optional, Tuple, Union

import math
import torch

from diffusers.utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler


def ddim_step_KL_search(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    old_model_output: torch.FloatTensor,
    timestep: torch.Tensor,   # now a tensor of shape (B,) of indices
    sample: torch.FloatTensor,  # shape: (B, C, H, W)
    eta: float = 0.0,
    duplicate: int = 1, 
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    timestep = timestep.long()
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )
    # 1. Get previous step value (element-wise)
    # Note: self.config.num_train_timesteps//self.num_inference_steps is an int.
    step_offset = self.config.num_train_timesteps // self.num_inference_steps
    prev_timestep = timestep - step_offset  # (B,) tensor

    # 2. Compute alphas, betas using advanced indexing.
    # We assume self.alphas_cumprod is a tensor and supports indexing with a tensor.
    alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]  # shape: (B,)
    # For prev_timestep, if any element is negative, we replace it with final_alpha_cumprod.
    # Create a mask and use torch.where:
    mask = (prev_timestep >= 0).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        mask, 
        self.alphas_cumprod.to(timestep.device)[prev_timestep],  # valid entries
        self.final_alpha_cumprod.expand_as(timestep).to(timestep.device)  # broadcast final_alpha_cumprod for invalid ones
    )
    beta_prod_t = 1 - alpha_prod_t  # (B,)

    # Unsqueeze to shape (B,1,1,1) for broadcasting with sample.
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3. Compute predicted original sample (x_0) and epsilon.
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output

        old_pred_original_sample = (sample - beta_prod_t.sqrt() * old_model_output) / alpha_prod_t.sqrt()
        old_pred_epsilon = old_model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be `epsilon`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
        old_pred_original_sample = self._threshold_sample(old_pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
        old_pred_original_sample = old_pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 5. Compute variance: sigma_t.

    variance = ((beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)).view(-1, 1, 1, 1)
    variance_coeff = variance

    std_dev_t = eta * variance.sqrt()

    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
        old_pred_epsilon = (sample - alpha_prod_t.sqrt() * old_pred_original_sample) / beta_prod_t.sqrt()

    # 6. Compute "direction pointing to x_t"
    # We need to unsqueeze alpha_prod_t_prev similarly.
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_epsilon
    old_pred_epsilon = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * old_pred_epsilon

    # 7. Compute x_t-1 (prev_sample) without noise.
    prev_sample_mean = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
    old_prev_sample_mean = alpha_prod_t_prev.sqrt() * old_pred_original_sample + old_pred_epsilon
    if eta > 0:
        # We assume generator and variance_noise handling remain similar.
        if variance_noise is not None and generator is not None:
            raise ValueError(
                "Cannot pass both generator and variance_noise. Please ensure only one is provided."
            )
        if variance_noise is None:
            model_shape = model_output.shape  # (B, C, H, W)
            variance_noise = randn_tensor(
                (model_shape[0] * duplicate, model_shape[1], model_shape[2], model_shape[3]),
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype
            )
        variance = std_dev_t.repeat_interleave(duplicate, dim=0) * variance_noise  
        # Here, we assume that for each sample in the batch, duplicate candidates are produced.
        # We'll adjust the addition accordingly:
        # First, we need to repeat prev_sample_mean along batch dimension duplicate times.
        prev_sample_mean_repeated = prev_sample_mean.repeat_interleave(duplicate, dim=0)  # shape: (B*duplicate, C, H, W)
        prev_sample = prev_sample_mean_repeated + variance

        kl_terms = (prev_sample_mean - old_prev_sample_mean)**2 / (2 * (std_dev_t**2))
        kl_terms = kl_terms.mean(dim=tuple(range(1, kl_terms.ndim)))

    else:
        prev_sample = prev_sample_mean
        kl_terms = torch.zeros(prev_sample_mean.size(0), device=prev_sample_mean.device)
        
    log_probs = (
        -((prev_sample.detach() - prev_sample_mean_repeated) ** 2) / (2 * ((std_dev_t.repeat_interleave(duplicate, dim=0))**2))
        - torch.log(std_dev_t.repeat_interleave(duplicate, dim=0))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    
    log_probs = log_probs.mean(dim=tuple(range(1, log_probs.ndim)))

    return prev_sample.to(dtype=sample.dtype), pred_original_sample, variance_coeff, variance, kl_terms, log_probs


def predict_x0_from_xt_search(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.FloatTensor] = None,
) -> Union[DDIMSchedulerOutput, Tuple]:
    timestep = timestep.long()
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    
    # 2. Compute alphas, betas using advanced indexing.
    # We assume self.alphas_cumprod is a tensor and supports indexing with a tensor.
    alpha_prod_t = self.alphas_cumprod.to(timestep.device)[timestep]  # shape: (B,)
    # For prev_timestep, if any element is negative, we replace it with final_alpha_cumprod.
    # Create a mask and use torch.where:
    mask = (prev_timestep >= 0).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        mask, 
        self.alphas_cumprod.to(timestep.device)[prev_timestep],  # valid entries
        self.final_alpha_cumprod.expand_as(timestep).to(timestep.device)  # broadcast final_alpha_cumprod for invalid ones
    )
    beta_prod_t = 1 - alpha_prod_t
    
    alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
    alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1)
    beta_prod_t = beta_prod_t.view(-1, 1, 1, 1)
    
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon": # default
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be `epsilon`"
        )

    # 4. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
    
    return pred_original_sample.to(dtype=sample.dtype)
