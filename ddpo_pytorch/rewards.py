from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score(dtype = torch.float32):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype).cuda()

    def _fn(images, prompts, metadata=None):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def aesthetic_score_diff(torch_dtype=torch.float32):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorerDiff
    
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(dtype=torch_dtype)
    scorer.requires_grad_(False)
    
    def loss_fn(im_pix, prompts=None, metadata=None):
        if im_pix.min() < 0:
            im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        im_pix = im_pix.to(torch_dtype)
        scorer_ = scorer.to(im_pix.device)
        rewards = scorer_(im_pix)
        return rewards, rewards
    return loss_fn

