import os
import re
import gc
import csv
import math
import argparse
import random
import warnings
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
import matplotlib.pyplot as plt

import eval_reward as rewards
from dreamsim import dreamsim


# ───────────────────────── GLOBAL ─────────────────────────
# GPU 설정은 main 함수에서 처리
DTYPE_IMG = torch.float32

TFM_LPIPS = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ───────────────────────── UTIL ─────────────────────────
def mean_and_se(vals: List[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if len(vals) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def collect_folders(base_dir: str, every: int):
    """
    새로운 폴더 구조에 맞게 수정된 폴더 수집 로직
    - base_dir 자체가 experiment 폴더임
    - base_dir 안에 checkpoint_* 폴더들이 있음
    """
    # base_dir에서 직접 checkpoint 폴더들을 찾기
    checkpoints_in_exp = []
    all_checkpoints = []
    
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        
        # checkpoint_<num> 패턴 매칭
        m = re.match(r"checkpoint_(\d+)$", name)
        if m:
            checkpoint_num = int(m.group(1))
            checkpoints_in_exp.append(checkpoint_num)
            if checkpoint_num % every == 0:  # every 간격으로만 수집
                all_checkpoints.append((name, checkpoint_num))
    
    print(f"Found checkpoints: {sorted(checkpoints_in_exp)}")
    print(f"Selected checkpoints: {[num for _, num in all_checkpoints]}")
    
    # checkpoint 번호로 정렬
    all_checkpoints.sort(key=lambda x: x[1])
    
    # 단일 experiment로 그룹화
    experiments_dict = {"experiment": all_checkpoints}
    
    print(f"Total checkpoints to evaluate: {len(all_checkpoints)}")
    return experiments_dict, all_checkpoints


def iter_images(folder: str) -> Iterable[str]:
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "ess" not in f and "intermediate_rewards" not in f:
            yield os.path.join(folder, f)


def parse_aesthetic(fname: str) -> Optional[float]:
    """
    파일명 끝의 '_{float}.jpg' 패턴에서 float 점수를 추출
    예: 'checkpoint_19_eval_159_camel_7.0750.jpg' -> 7.0750
    """
    m = re.search(r"_([-+]?[0-9]*\.?[0-9]+)\.(png|jpg|jpeg)$", fname, re.I)
    return float(m.group(1)) if m else None


def identify_prompt_from_name(fname: str, prompts: List[str], truncated: bool = False) -> Optional[str]:
    """
    파일명에서 prompt 추출.
    새로운 파일명 패턴: checkpoint_19_eval_159_camel_7.0750.jpg
    - camel 부분이 prompt
    """
    name_noext = os.path.splitext(fname)[0]
    
    # checkpoint_19_eval_159_camel_7.0750 패턴에서 camel 추출
    parts = name_noext.split('_')
    if len(parts) >= 4:
        # checkpoint_19_eval_159_camel_7.0750 -> camel
        prompt_part = parts[3]
        
        # prompts 리스트에서 가장 잘 맞는 것 찾기
        for p in sorted(prompts, key=len, reverse=True):
            if p.lower() == prompt_part.lower():
                return p
        
        # 정확히 일치하지 않으면 부분 일치 시도
        for p in sorted(prompts, key=len, reverse=True):
            if p.lower() in prompt_part.lower() or prompt_part.lower() in p.lower():
                return p
    
    return None


def load_prompts(prompt_path: str) -> List[str]:
    with open(prompt_path, encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    prompts.sort(key=len, reverse=True)
    return prompts


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
def load_images(folder: str, prompts_list: List[str], truncated: bool = False):
    paths, pils, prompts, aesthetics = [], [], [], []
    to_tensor = torchvision.transforms.ToTensor()

    print(f"  Loading images from {folder}...")
    for p in iter_images(folder):
        paths.append(p)
        pil = Image.open(p).convert("RGB")
        pils.append(pil)
        prompts.append(identify_prompt_from_name(os.path.basename(p), prompts_list, truncated=truncated))
        aesthetics.append(parse_aesthetic(os.path.basename(p)))

    if not pils:
        return {"paths": [], "pil": [], "gpu": None, "prompts": [], "aesthetics": []}

    imgs_cpu = torch.stack([to_tensor(im) for im in pils], dim=0).contiguous()  # CPU only

    return {"paths": paths, "pil": pils, "cpu": imgs_cpu, "prompts": prompts, "aesthetics": aesthetics}


def evaluate_score_metrics(folder: str, imgs: Dict, bundle: ModelBundle, batch_size: int = 128):
    B = batch_size
    results_dir = os.path.join(folder, "results")
    os.makedirs(results_dir, exist_ok=True)

    x_cpu = imgs["cpu"]
    if x_cpu is None:
        print(f"  No images found in {folder}")
        return {}
        
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
            # None 값이 있는 prompt는 건너뛰기
            valid_indices = [i for i, p in enumerate(prb) if p is not None]
            if not valid_indices:
                continue
                
            valid_xb = xb[valid_indices]
            valid_prb = [prb[i] for i in valid_indices]
            
            out = bundle.scorers[name](valid_xb, valid_prb)
            scores[name].extend(_to_list(out))

        del xb  # 배치 끝낼 때 즉시 해제

    row = {}
    for k, v in scores.items():
        m, s = mean_and_se(v)
        row[f"mean_{k}"], row[f"std_{k}"] = m, s

    return row


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


def evaluate_diversity_metrics(folder: str, imgs: Dict, bundle: ModelBundle, args):
    """
    한 번의 모델 로드로:
      - CLIP 임베딩 전체 추출 → CPU 저장 → whole & per-prompt 계산
      - DreamSim 임베딩 전체 추출 → CPU 저장 → whole & per-prompt 계산
      - LPIPS 전처리 텐서 전체 준비(CPU) → 필요쌍만 GPU 전송 → whole & per-prompt 계산
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

    # -------------------- Return results --------------------
    return {
        # whole
        "clip_diversity": clip_div_mean, "clip_diversity_se": clip_div_se,
        "dreamsim_diversity": dream_div_mean, "dreamsim_diversity_se": dream_div_se,
        "lpips_mean": lpips_mean, "lpips_se": lpips_se,
        # per-prompt
        "clip_diversity_prompt": clip_div_pp_mean, "clip_diversity_prompt_se": clip_div_pp_se,
        "dreamsim_diversity_prompt": dream_div_pp_mean, "dreamsim_diversity_prompt_se": dream_div_pp_se,
        "lpips_diversity_prompt": lpips_div_pp_mean, "lpips_diversity_prompt_se": lpips_div_pp_se,
    }


def evaluate_folder(folder: str, args, bundle: ModelBundle) -> Dict[str, float]:
    """
    한 폴더를 평가하여 모든 메트릭을 계산
    """
    print(f"\nEvaluating folder: {folder}")
    
    # 이미지 로드
    prompts = load_prompts(args.prompt_dir)
    imgs = load_images(folder, prompts, args.truncated_prompt)
    
    if not imgs["pil"]:
        print(f"  No images found in {folder}")
        return {}
    
    # 품질 메트릭 평가
    score_results = evaluate_score_metrics(folder, imgs, bundle, args.batch_size)
    
    # 다양성 메트릭 평가
    diversity_results = evaluate_diversity_metrics(folder, imgs, bundle, args)
    
    # 결과 병합
    results = {**score_results, **diversity_results}
    
    # 기존 eval.py와 호환되는 CSV 저장
    with open(os.path.join(folder, "eval_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())
        writer.writerow([f"{v:.5f}" if isinstance(v, float) and not math.isnan(v) else str(v) for v in results.values()])
    
    # 메모리 정리
    del imgs
    torch.cuda.empty_cache()
    
    return results


def run(base_dir: str, args):
    """
    새로운 폴더 구조에 맞게 수정된 run 함수
    """
    out_dir = os.path.join(base_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    
    experiments_dict, all_checkpoints = collect_folders(base_dir, args.every)
    
    # 모델 번들 로드 (한 번만)
    bundle = ModelBundle(device=DEVICE, inference_dtype=torch.float32)
    
    results = {}
    checkpoint_numbers = sorted(set([num for _, num in all_checkpoints]))
    
    # 각 experiment별로 평가
    for exp_name, checkpoints in experiments_dict.items():
        print(f"\n=== Evaluating experiment: {exp_name} ===")
        exp_results = {}
        
        for checkpoint_name, checkpoint_num in tqdm(checkpoints, desc=f"Evaluating {exp_name}"):
            try:
                folder_path = os.path.join(base_dir, checkpoint_name)
                result = evaluate_folder(folder_path, args, bundle)
                if result:  # 결과가 비어있지 않은 경우만 저장
                    exp_results[checkpoint_num] = result
                    print(f"  ✓ {checkpoint_name}: {len(result)} metrics")
                else:
                    print(f"  ✗ {checkpoint_name}: No results")
            except Exception as e:
                import traceback
                print(f"[!] {exp_name}/{checkpoint_name} skipped ({e})")
                print(f"    Traceback: {traceback.format_exc()}")
        
        results[exp_name] = exp_results
        print(f"  Total results for {exp_name}: {len(exp_results)}")
    
    print(f"\nTotal experiments with results: {len([r for r in results.values() if r])}")
    for exp_name, exp_results in results.items():
        print(f"  {exp_name}: {len(exp_results)} checkpoints")
    
    # CSV 저장 - experiment별로
    for exp_name, exp_results in results.items():
        if not exp_results:
            continue
            
        csv_path = os.path.join(out_dir, f"{exp_name}_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            hdr = ["checkpoint"] + list(next(iter(exp_results.values())).keys()) if exp_results else []
            csv.writer(f).writerow(hdr)
            for checkpoint_num in sorted(exp_results.keys()):
                if checkpoint_num in exp_results:
                    csv.writer(f).writerow([checkpoint_num] + [f"{exp_results[checkpoint_num][k]:.5f}" if isinstance(exp_results[checkpoint_num][k], float) and not math.isnan(exp_results[checkpoint_num][k]) else str(exp_results[checkpoint_num][k]) for k in hdr[1:]])
        print(f"✓ saved {csv_path}")
    
    # 전체 통합 CSV
    all_csv_path = os.path.join(out_dir, "all_experiments_metrics.csv")
    with open(all_csv_path, "w", newline="") as f:
        if results:
            # 첫 번째 experiment의 첫 번째 결과에서 헤더 가져오기
            first_exp = next(iter(results.values()))
            if first_exp:
                first_result = next(iter(first_exp.values()))
                hdr = ["experiment", "checkpoint"] + list(first_result.keys())
                csv.writer(f).writerow(hdr)
                
                for exp_name, exp_results in results.items():
                    for checkpoint_num in sorted(exp_results.keys()):
                        if checkpoint_num in exp_results:
                            csv.writer(f).writerow([exp_name, checkpoint_num] + [f"{exp_results[checkpoint_num][k]:.5f}" if isinstance(exp_results[checkpoint_num][k], float) and not math.isnan(exp_results[checkpoint_num][k]) else str(exp_results[checkpoint_num][k]) for k in first_result.keys()])
    print(f"✓ saved {all_csv_path}")
    
    # plotting - experiment별로
    if results:
        metrics = [
            ("mean_aesthetic","std_aesthetic","Aesthetic","aesthetic"),
            ("mean_hps","std_hps","HPS","hps"),
            ("mean_imagereward","std_imagereward","ImageReward","imagereward"),
            ("mean_pick","std_pick","Pick","pick"),
            ("mean_clip","std_clip","CLIP","clip"),
            ("clip_diversity","clip_diversity_se","CLIP-Div","clip_diversity"),
            ("dreamsim_diversity","dreamsim_diversity_se","DreamSim-Div","dreamsim_diversity"),
            ("lpips_mean","lpips_se","LPIPS-Mean","lpips_mean"),
            ("clip_diversity_prompt", "clip_diversity_prompt_se", "CLIP-Div (prompt)", "clip_div_prompt"),
            ("dreamsim_diversity_prompt", "dreamsim_diversity_prompt_se", "DreamSim-Div (prompt)", "dreamsim_div_prompt"),
            ("lpips_diversity_prompt","lpips_diversity_prompt_se","LPIPS-Div (prompt)","lpips_div_prompt"),
        ]

        for key,std_key,label,suf in metrics:
            plt.figure(figsize=(12,8))
            
            for exp_name, exp_results in results.items():
                if not exp_results:
                    continue
                    
                checkpoints = sorted(exp_results.keys())
                values = [exp_results[c].get(key, np.nan) for c in checkpoints]
                stds = [exp_results[c].get(std_key, 0.0) for c in checkpoints] if std_key else [0.0]*len(checkpoints)
                
                plt.plot(checkpoints, values, 'o-', label=exp_name, linewidth=2, markersize=6)
                plt.fill_between(checkpoints, 
                               np.array(values)-np.array(stds), 
                               np.array(values)+np.array(stds), 
                               alpha=.2)

            plt.xlabel(f"Checkpoint (every {args.every})"); plt.ylabel(label)
            plt.title(f"{label} across experiments"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
            out_png = os.path.join(out_dir,f"metrics_{suf}.png")
            plt.savefig(out_png,dpi=200); plt.close(); print("✓",out_png)


def main(args):
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using GPU {args.gpu_id} ({DEVICE})")
    
    with torch.inference_mode():
        run(args.base_dir, args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description="Evaluate eval/search folders using new evaluation logic.")
    parser.add_argument("--base_dir", required=True, help="Directory containing experiment folders with checkpoint_* subfolders")
    parser.add_argument("--prompt_dir", type=str, default="./assets/simple_animals.txt", help="Path to prompt text file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--every", type=int, default=2, help="Stride for epochs/steps (default 3)")
    parser.add_argument("--num_random_images", type=int, default=4, help="Number of random images per prompt for diversity")
    parser.add_argument("--K", type=int, default=5, help="Number of repeats for diversity evaluation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--truncated_prompt", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Whether prompts are truncated in filenames")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    
    fix_seed(args.seed)
    main(args)
