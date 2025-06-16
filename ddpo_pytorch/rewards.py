from PIL import Image
import io
import numpy as np
import torch
import torchvision
import ImageReward as RM
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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

def aesthetic_score_diff_clipped(torch_dtype=torch.float32):
    from ddpo_pytorch.aesthetic_scorer import AestheticScorerDiff
    
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(dtype=torch_dtype)
    scorer.requires_grad_(False)
    
    def loss_fn(im_pix, prompts=None, metadata=None):
        if im_pix.min() < 0:
            im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        im_pix = im_pix.to(torch_dtype)
        scorer_ = scorer.to(im_pix.device)
        rewards = scorer_(im_pix)
        return -torch.abs(8.5-rewards), -torch.abs(8.5-rewards)
    return loss_fn






def clip_score(
    return_loss=False, 
):
    from ddpo_pytorch.clip_scorer import CLIPScorer

    scorer = CLIPScorer(dtype=torch.float32, device='cuda')
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





# def hps_score(
#     return_loss=False, 
# ):
#     from ddpo_pytorch.hpsv2_scorer import HPSv2Scorer

#     scorer = HPSv2Scorer(dtype=torch.float32, device='cuda').cuda()
#     scorer.requires_grad_(False)

#     if not return_loss:
#         def _fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)
#             return scores

#         return _fn

#     else:
#         def loss_fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)

#             loss = 1.0 - scores
#             return loss, scores

#         return loss_fn


# def ImageReward(
#     return_loss=False, 
# ):
#     from ddpo_pytorch.ImageReward_scorer import ImageRewardScorer

#     scorer = ImageRewardScorer(dtype=torch.float32, device='cuda').cuda()
#     scorer.requires_grad_(False)

#     if not return_loss:
#         def _fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)
#             return scores

#         return _fn

#     else:
#         def loss_fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)

#             loss = - scores
#             return loss, scores

#         return loss_fn
    


# def PickScore(
#     return_loss=False, 
# ):
#     from ddpo_pytorch.PickScore_scorer import PickScoreScorer

#     scorer = PickScoreScorer(dtype=torch.float32, device='cuda').cuda()
#     scorer.requires_grad_(False)

#     if not return_loss:
#         def _fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)
#             return scores

#         return _fn

#     else:
#         def loss_fn(images, prompts):
#             if images.min() < 0: # normalize unnormalized images
#                 images = ((images / 2) + 0.5).clamp(0, 1)
#             scores = scorer(images, prompts)

#             loss = - scores
#             return loss, scores

#         return loss_fn

def multi_reward_evaluation():
    def _fn(images, prompts):
        aesthetic_score_fn = aesthetic_score()
        # hps_score_fn = hps_score()
        # pick_score_fn = PickScore()
        # image_reward_fn = ImageReward()
        # clip_score_fn = clip_score()
        return {
            # "aesthetic_score": aesthetic_score_fn(images, prompts),
            # "hps_score": hps_score_fn(images, prompts),
            # "PickScore": pick_score_fn(images, prompts),
            # "ImageReward": image_reward_fn(images, prompts),
            # "clip_score": clip_score_fn(images, prompts)
        }
    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn



def hps_score(
    inference_dtype=None, 
    device='cuda', 
    return_loss=False, 
):
    from ddpo_pytorch.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts, metadata=None):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)
            return scores, {}

        return _fn

    else:
        def loss_fn(images, prompts):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts, metadata=None)

            loss = 1.0 - scores
            return loss, scores

        return loss_fn


def ImageReward(dtype=torch.float32, device="cuda", accelerator=None):
    
    # def get_local_rank():
    #     return int(os.environ.get('LOCAL_RANK', '0'))

    # if get_local_rank() == 0:  # only download once
    #     reward_model = RM.load("ImageReward-v1.0")

    # dist.barrier()
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.to(dtype).to(device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts, meta=None):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=reward_model.blip.tokenizer.model_max_length) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images.to(dtype)))
        reward = reward.reshape(images.shape[0]).float()  # bf16 -> f32
        # reward = F.relu(reward)

        # 4) loss = 1 - reward
        loss = -1 * torch.nn.functional.relu(reward)

        return reward, loss  # differentiable

    return _fn


def PickScore(
    inference_dtype=None, 
    device='cuda', 
    return_loss=False, 
):
    from ddpo_pytorch.PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    if not return_loss:
        def _fn(images, prompts, metadata=None):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scorer_ = scorer.to(dtype=inference_dtype).to(images.device)
            scores = scorer_(images, prompts)
            return scores, {}

        return _fn

    else:
        def loss_fn(images, prompts, metadata=None):
            if images.min() < 0: # normalize unnormalized images
                images = ((images / 2) + 0.5).clamp(0, 1)
            scores = scorer(images, prompts)

            loss = - scores
            return scores, loss

        return loss_fn