import os
import re
import gc
import csv
import math
import argparse
import random
from typing import List, Dict, Tuple, Iterable, Optional
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy.spatial.distance import pdist
from itertools import combinations
from tqdm import tqdm
import lpips
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

import eval_reward as rewards
from dreamsim import dreamsim


# ───────────────────────── GLOBAL ─────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_IMG = torch.float32

TFM_LPIPS = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ───────────────────────── UTIL ─────────────────────────
def mean_and_se(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if len(vals) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def list_epoch_folders(exp_dir: str, every: int) -> List[Tuple[int, str]]:
    """
    exp_dir = 'eval/{exp_name}'
    반환: [(epoch_num, abs_path), ...] with epoch % every == 0
    """
    out = []
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Not found: {exp_dir}")
    for name in os.listdir(exp_dir):
        m = re.fullmatch(r"epoch(\d+)", name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep % every == 0:
            out.append((ep, os.path.join(exp_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def iter_images(folder: str) -> Iterable[str]:
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "ess" not in f and "intermediate_rewards" not in f:
            yield os.path.join(folder, f)


def parse_aesthetic(fname: str) -> Optional[float]:
    """
    파일명 끝의 '_{float}.png' 패턴에서 float 점수를 추출
    예: 'ant_0_5.482707.png' -> 5.482707
    """
    m = re.search(r"_([-+]?[0-9]*\.?[0-9]+)\.(png|jpg|jpeg)$", fname, re.I)
    return float(m.group(1)) if m else None


def check_eval_folder(args):
    """
    eval/{exp_name}/epoch{num} 폴더들을 훑으면서,
    각 epoch 폴더의 이미지 총 개수가
    (프롬프트 수 × num_images_per_prompt) 와 일치하는지 확인.
    """
    exp_dir = os.path.join("eval", args.exp_name)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Not found: {exp_dir}")

    # 프롬프트 수 계산
    with open(args.prompt_dir, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    num_prompts = len(prompts)
    expected = num_prompts * args.num_images_per_prompt

    # epoch 폴더 수집 (every 간격만)
    epoch_dirs = []
    for name in os.listdir(exp_dir):
        m = re.fullmatch(r"epoch(\d+)", name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep % args.every == 0:
            epoch_dirs.append((ep, os.path.join(exp_dir, name)))
    epoch_dirs.sort(key=lambda x: x[0])

    if not epoch_dirs:
        print(f"[check_eval_folder] 검사 대상 epoch 폴더가 없습니다. (every={args.every})")
        return

    any_issue = False
    for ep, ep_dir in epoch_dirs:
        cnt = 0
        for entry in os.scandir(ep_dir):
            if not entry.is_file():
                continue
            fn = entry.name.lower()
            if (fn.endswith((".png", ".jpg", ".jpeg"))
                and "ess" not in fn
                and "intermediate_rewards" not in fn):
                cnt += 1

        if cnt != expected:
            any_issue = True
            diff = expected - cnt
            sign = "-" if diff > 0 else "+"
            print(f"[epoch{ep}] 개수 불일치: {cnt} / {expected} (차이 {sign}{abs(diff)})")
        else:
            print(f"[epoch{ep}] ✓ OK ({cnt} files)")

    if not any_issue:
        print("✓ 모든 대상 epoch 폴더가 기대 개수와 일치합니다.")


# ───────────────────────── MODELS (LOAD ONCE) ─────────────────────────
def _freeze(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    m.eval()

class ModelBundle:
    def __init__(self, device: str, inference_dtype: torch.dtype = torch.float32):
        self.device = device
        self.inference_dtype = inference_dtype

        print("[ModelBundle] Loading quality scorers...")
        self.scorers = {
            "clip":        rewards.clip_score(inference_dtype=inference_dtype, device=device),
            "hps":         rewards.hps_score(inference_dtype=inference_dtype, device=device),
            "imagereward": rewards.ImageReward(inference_dtype=inference_dtype, device=device),
            "pick":        rewards.PickScore(inference_dtype=inference_dtype, device=device),
        }

        for s in self.scorers.values():
            if isinstance(s, torch.nn.Module):
                _freeze(s)

        print("[ModelBundle] Loading CLIP (for embeddings)...")
        self.clip_emb_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        _freeze(self.clip_emb_model)
        self.clip_emb_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        print("[ModelBundle] Loading DreamSim...")
        self.dream_model, self.dream_pre = dreamsim(pretrained=True, device=device)
        if isinstance(self.dream_model, torch.nn.Module):
            _freeze(self.dream_model)

        print("[ModelBundle] Loading LPIPS...")
        self.lpips_model = lpips.LPIPS(net="alex").to(device)
        _freeze(self.lpips_model)

        torch.cuda.empty_cache(); gc.collect()

# ───────────────────────── EVAL (REUSE MODELS) ─────────────────────────
def evaluate_score_metrics(args, epoch: int, imgs: Dict, bundle: ModelBundle):
    B = args.batch_size
    exp_dir = os.path.join("eval", args.exp_name)
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    x_cpu = imgs["cpu"]                         
    img_prompts = imgs["prompts"]
    N = x_cpu.shape[0]

    scores = {
        "aesthetic": [a for a in imgs["aesthetics"] if a is not None],
        "clip": [], "hps": [], "imagereward": [], "pick": [],
    }

    def _to_list(out):
        if isinstance(out, torch.Tensor):
            return out.detach().float().cpu().tolist()
        elif isinstance(out, np.ndarray):
            return out.astype(np.float32).tolist()
        else:
            return list(map(float, out))

    for i in tqdm(range(0, N, B), desc="Evaluating score metrics"):
        sl = slice(i, i + B)
        xb = x_cpu[sl].to(DEVICE)   
        prb = img_prompts[sl] if isinstance(img_prompts, list) else img_prompts

        for name in ("hps", "imagereward", "pick", "clip"):
            out = bundle.scorers[name](xb, prb)
            scores[name].extend(_to_list(out))

        del xb  # 배치 끝낼 때 즉시 해제

    row = {}
    for k, v in scores.items():
        m, s = mean_and_se(v)
        row[f"mean_{k}"], row[f"std_{k}"] = m, s

    out_csv = os.path.join(results_dir, f"score_epoch{epoch}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["epoch"] + list(row.keys())
        w.writerow(hdr)
        w.writerow([epoch] + [f"{row[k]:.5f}" if isinstance(row[k], float) and not math.isnan(row[k]) else str(row[k]) for k in row])
    print(f"✓ saved {out_csv}")

def _lpips_pairwise_mean(lpips_model, tensors: List[torch.Tensor], device: str, pair_batch: int = 64) -> float:
    n = len(tensors)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    vals = []
    with torch.no_grad():
        for s in range(0, len(pairs), pair_batch):
            chunk = pairs[s:s+pair_batch]
            a_batch = torch.cat([tensors[i] for i, _ in chunk], dim=0).to(device)  # [B,3,224,224]
            b_batch = torch.cat([tensors[j] for _, j in chunk], dim=0).to(device)

            d = lpips_model(a_batch, b_batch).squeeze().detach().float().cpu().numpy().ravel().tolist()

            vals.extend(d)
            del a_batch, b_batch

    return float(np.mean(vals)) if vals else float("nan")


def build_stratified_index_sets(
    prompt_to_idx: Dict[str, List[int]],
    k_per_prompt: int,
    repeats: int,
    seed: int,
) -> List[np.ndarray]:
    """
    프롬프트별로 k개씩 뽑아 합친 인덱스 집합을 K회(repeats) 생성하고 리스트로 반환.
    - 프롬프트 key는 정렬하여 순서 안정화
    - 각 repeat마다 prompt 내에서는 중복 없이 sample
    - 서로 다른 repeat 간에는 중복될 수 있음(의도)
    """
    rng = random.Random(seed)
    pr_keys = sorted([pr for pr in prompt_to_idx.keys() if prompt_to_idx[pr]])
    sets = []
    for _ in range(repeats):
        sampled = []
        for pr in pr_keys:
            lst = prompt_to_idx[pr]
            if len(lst) < k_per_prompt:
                raise ValueError(f"Prompt '{pr}' has only {len(lst)} images (< {k_per_prompt}).")
            sampled += rng.sample(lst, k_per_prompt)
        sets.append(np.asarray(sampled, dtype=np.int32))
    return sets


def evaluate_diversity_metrics(args, ep: int, imgs: Dict, bundle: ModelBundle):
    """
    한 번의 모델 로드로:
      - CLIP 임베딩 전체 추출 → CPU 저장 → whole & per-prompt 계산
      - DreamSim 임베딩 전체 추출 → CPU 저장 → whole & per-prompt 계산
      - LPIPS 전처리 텐서 전체 준비(CPU) → 필요쌍만 GPU 전송 → whole & per-prompt 계산
    CSV: results/div_epoch{ep}.csv
    """
    # prompt -> indices
    prompt_to_idx: Dict[str, List[int]] = defaultdict(list)
    for idx, pr in enumerate(imgs["prompts"]):
        if pr is not None:
            prompt_to_idx[pr].append(idx)

    # stratified index sets
    whole_sets = build_stratified_index_sets(
        prompt_to_idx=prompt_to_idx,
        k_per_prompt=args.num_random_images,
        repeats=args.K,
        seed=args.seed,
    )

    B = args.batch_size

    # -------------------- CLIP --------------------
    clip_embs_cpu = []
    print("  CLIP embedding start...")
    for i in range(0, len(imgs["pil"]), B):
        batch_pil = imgs["pil"][i:i+B]
        pixel = bundle.clip_emb_proc(images=batch_pil, return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.no_grad():
            e = bundle.clip_emb_model.get_image_features(pixel).float().cpu().numpy()
        clip_embs_cpu.append(e)
    clip_embs_cpu = np.concatenate(clip_embs_cpu, axis=0)  # [N, D]

    print("  CLIP whole evaluation start...")
    clip_whole_vals = [float(pdist(clip_embs_cpu[idxs], metric="cosine").mean()) for idxs in whole_sets]

    # per-prompt
    clip_pp_vals = []
    print("  CLIP per-prompt evaluation start...")
    for pr in sorted(prompt_to_idx.keys()):
        idxs = prompt_to_idx[pr]
        if not idxs or len(idxs) < 2:
            raise ValueError(f"No images or not enough images found for prompt: {pr}")
        clip_pp_vals.append(float(pdist(clip_embs_cpu[idxs], metric="cosine").mean()))

    clip_div_mean, clip_div_se = mean_and_se(clip_whole_vals)
    clip_div_pp_mean, clip_div_pp_se = mean_and_se(clip_pp_vals)

    # -------------------- DreamSim --------------------
    dream_embs_cpu = []
    print("  DreamSim embedding start...")
    for i in range(0, len(imgs["pil"]), B):
        batch_pil = imgs["pil"][i:i+B]
        pxs = torch.cat([bundle.dream_pre(im) for im in batch_pil], dim=0).to(DEVICE)
        with torch.no_grad():
            e = bundle.dream_model.embed(pxs).float().cpu().numpy()
        dream_embs_cpu.append(e)
    dream_embs_cpu = np.concatenate(dream_embs_cpu, axis=0)  # [N, D']

    print("  DreamSim whole evaluation start...")
    dream_whole_vals = [float(pdist(dream_embs_cpu[idxs], metric="cosine").mean()) for idxs in whole_sets]

    dream_pp_vals = []
    print("  DreamSim per-prompt evaluation start...")
    for pr in sorted(prompt_to_idx.keys()):
        idxs = prompt_to_idx[pr]
        if not idxs or len(idxs) < 2:
            raise ValueError(f"No images or not enough images found for prompt: {pr}")
        dream_pp_vals.append(float(pdist(dream_embs_cpu[idxs], metric="cosine").mean()))

    dream_div_mean, dream_div_se = mean_and_se(dream_whole_vals)
    dream_div_pp_mean, dream_div_pp_se = mean_and_se(dream_pp_vals)

    # -------------------- LPIPS --------------------
    lpips_tensors_cpu = [TFM_LPIPS(pil).unsqueeze(0).contiguous() for pil in imgs["pil"]]  # list of [1,3,224,224] CPU

    lpips_whole_vals = []
    print("  LPIPS whole evaluation start...")
    for idxs in whole_sets:
        tensors = [lpips_tensors_cpu[int(j)] for j in idxs]
        lpips_whole_vals.append(_lpips_pairwise_mean(bundle.lpips_model, tensors, DEVICE, pair_batch=B))

    lpips_pp_vals = []
    print("  LPIPS per-prompt evaluation start...")
    for pr in sorted(prompt_to_idx.keys()):
        idxs = prompt_to_idx[pr]
        if not idxs or len(idxs) < 2:
            raise ValueError(f"No images or not enough images found for prompt: {pr}")
        tensors = [lpips_tensors_cpu[j] for j in idxs]
        lpips_pp_vals.append(_lpips_pairwise_mean(bundle.lpips_model, tensors, DEVICE, pair_batch=B))

    lpips_mean, lpips_se = mean_and_se(lpips_whole_vals)
    lpips_div_pp_mean, lpips_div_pp_se = mean_and_se(lpips_pp_vals)

    # -------------------- Save one CSV --------------------
    row = {
        "epoch": ep,
        # whole
        "clip_div_mean": clip_div_mean, "clip_div_se": clip_div_se,
        "dreamsim_div_mean": dream_div_mean, "dreamsim_div_se": dream_div_se,
        "lpips_mean": lpips_mean, "lpips_se": lpips_se,
        # per-prompt
        "clip_div_pp_mean": clip_div_pp_mean, "clip_div_pp_se": clip_div_pp_se,
        "dreamsim_div_pp_mean": dream_div_pp_mean, "dreamsim_div_pp_se": dream_div_pp_se,
        "lpips_div_pp_mean": lpips_div_pp_mean, "lpips_div_pp_se": lpips_div_pp_se,
    }
    exp_dir = os.path.join("eval", args.exp_name)
    results_dir = os.path.join(exp_dir, "results")
    out_csv = os.path.join(results_dir, f"div_epoch{ep}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); hdr = list(row.keys())
        w.writerow(hdr)
        w.writerow([row[k] if not isinstance(row[k], float) or math.isnan(row[k]) else f"{row[k]:.5f}" for k in hdr])
    print(f"✓ saved {out_csv}")


def process_csvs(args):
    """
    results/ 폴더 내 epoch별 CSV들을 병합하여 all_metrics_{exp_name}.csv 생성.
    - score_epoch*.csv : 품질 관련 metric (aesthetic, hps, imagereward, pick, clip)
    - div_epoch*.csv   : diversity 관련 metric (clip, dreamsim, lpips)
    """
    exp_dir = os.path.join("eval", args.exp_name)
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    def read_group(prefix: str) -> pd.DataFrame:
        rows = []
        for f in os.listdir(results_dir):
            if f.startswith(prefix) and f.endswith(".csv"):
                df = pd.read_csv(os.path.join(results_dir, f))
                rows.append(df)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    # 개별 그룹 읽기
    df_score = read_group("score_epoch")
    df_div   = read_group("div_epoch")

    # epoch 기준 병합
    df = None
    for part in [df_score, df_div]:
        if part.empty:
            continue
        df = part if df is None else pd.merge(df, part, on="epoch", how="outer")

    if df is None:
        print("결합할 CSV가 없습니다.")
        return

    df.sort_values("epoch", inplace=True)
    out_csv = os.path.join(results_dir, f"all_metrics_{args.exp_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"✓ merged CSV saved to {out_csv}")


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def identify_prompt_from_name(fname: str, prompts: List[str], truncated: bool = False) -> Optional[str]:
    """
    파일명에서 prompt 추출.
    truncated=False → 첫 번째 '_' 앞의 토큰을 prompt로 사용 (예: 'lion_0_5.23.png' → 'lion')
    truncated=True  → 파일명에 잘린 prompt가 들어있으므로 txt 목록에서 가장 맞는 prompt를 찾아 반환
    """
    name_noext = os.path.splitext(fname)[0]
    if not truncated:  # 첫 번째 '_' 이전이 prompt
        return name_noext.split("_")[0]

    key = name_noext.lower()
    for p in sorted(prompts, key=len, reverse=True):
        p_low = p.lower()
        if p_low in key or p_low.replace(" ", "_") in key:
            return p
    return None


def load_images(ep_dir: str, prompts_list: List[str], truncated: bool = False):
    paths, pils, prompts, aesthetics = [], [], [], []
    to_tensor = torchvision.transforms.ToTensor()

    print(f"  Loading images from {ep_dir}...")
    for p in iter_images(ep_dir):
        paths.append(p)
        pil = Image.open(p).convert("RGB")
        pils.append(pil)
        prompts.append(identify_prompt_from_name(os.path.basename(p), prompts_list, truncated=truncated))
        aesthetics.append(parse_aesthetic(os.path.basename(p)))

    if not pils:
        return {"paths": [], "pil": [], "gpu": None, "prompts": [], "aesthetics": []}

    imgs_cpu = torch.stack([to_tensor(im) for im in pils], dim=0).contiguous()  # CPU only

    return {"paths": paths, "pil": pils, "cpu": imgs_cpu, "prompts": prompts, "aesthetics": aesthetics}


def load_prompts(prompt_path: str) -> List[str]:
    with open(prompt_path, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    prompts.sort(key=len, reverse=True)
    return prompts


def main(args):
    with torch.inference_mode():
        exp_dir = os.path.join("eval", args.exp_name)
        prompts = load_prompts(args.prompt_dir)
        epochs = list_epoch_folders(exp_dir, args.every)

        # model load
        bundle = ModelBundle(device=DEVICE, inference_dtype=torch.float32)

        for ep, ep_dir in tqdm(epochs, desc="epoch", unit="epoch", unit_scale=1, position=0, leave=True):
            imgs = load_images(ep_dir, prompts, args.truncated_prompt)
            evaluate_score_metrics(args, ep, imgs, bundle)
            evaluate_diversity_metrics(args, ep, imgs, bundle)

            del imgs
            torch.cuda.empty_cache()

        process_csvs(args)

if __name__ == "__main__":
    '''
    folder structure 
    - eval
        - {exp_name}
            - epoch{epoch}
                - {prompt_name}_0_5.23.png
                - {prompt_name}_1_5.23.png
                - ...
            - epoch{epoch}
                - {prompt_name}_0_5.23.png
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="0719_195639")
    parser.add_argument("--prompt_dir", type=str, default="./assets/simple_animals.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_images_per_prompt", type=int, default=32)
    parser.add_argument("--every", type=int, default=40)
    parser.add_argument("--num_random_images", type=int, default=4)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--truncated_prompt", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    args = parser.parse_args()

    fix_seed(args.seed)

    check_eval_folder(args)
    main(args)