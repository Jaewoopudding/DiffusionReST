from PIL import Image
import os
from pathlib import Path
import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import pipeline
from diffusers.utils import load_image
from importlib import resources
ASSETS_PATH = resources.files("ddpo_pytorch.assets")

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(images):
        if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn

def clip_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from ddpo_pytorch.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn

def aesthetic_score(
    torch_dtype=None,
    aesthetic_target=None,
    grad_scale=0,
    device=None,
    return_loss=False,
):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images)

            if aesthetic_target is None: # default maximization
                loss = -1 * scores
            else:
                # using L1 to keep on same scale
                loss = abs(scores - aesthetic_target)
            return loss * grad_scale, scores

        return loss_fn


def hps_score(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from ddpo_pytorch.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn


def ImageReward(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from ddpo_pytorch.ImageReward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn
    


def PickScore(
    inference_dtype=None, 
    device=None, 
    return_loss=False, 
):
    from ddpo_pytorch.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return loss, scores

        return loss_fn