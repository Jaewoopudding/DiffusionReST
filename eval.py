import os, re, csv, math, warnings, argparse
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torchvision
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import lpips
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import eval_reward as rewards
from dreamsim import dreamsim


# ────────────────────────────── CONFIG ──────────────────────────────
PROMPT_TXT = "ddpo_pytorch/assets/simple_animals.txt"
with open(PROMPT_TXT, encoding="utf-8") as f:
    PROMPTS = [ln.strip() for ln in f if ln.strip()]
PROMPTS.sort(key=len, reverse=True)
 
device = "cuda" if torch.cuda.is_available() else "cpu"

hps_fn         = rewards.hps_score(inference_dtype=torch.float32, device=device)
imagereward_fn = rewards.ImageReward(inference_dtype=torch.float32, device=device)
pick_fn        = rewards.PickScore(inference_dtype=torch.float32, device=device)
clip_fn        = rewards.clip_score(inference_dtype=torch.float32, device=device)

clip_model  = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc   = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
lpips_model = lpips.LPIPS(net="alex").to(device)
dmodel, dpre = dreamsim(pretrained=True, device=device)

tfm_lpips = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ────────────────────────────── HELPERS ─────────────────────────────
def preprocess_clip(p): return clip_proc(images=Image.open(p).convert("RGB"),
                                         return_tensors="pt")["pixel_values"].squeeze(0)
def preprocess_lpips(p): return tfm_lpips(Image.open(p).convert("RGB")).unsqueeze(0)
def embed_dreamsim (p): return dmodel.embed(dpre(Image.open(p)).to(device)).detach()

def identify_prompt(fname: str) -> str:
    base = fname.lower()
    for tok in base.split("_"):
        if tok in PROMPTS: return tok
    for p in PROMPTS:
        if p in base:     return p
    raise ValueError(f"prompt not found in filename: {fname}")

def parse_aesthetic(fname: str):
    m = re.search(r"_([-+]?[0-9]*\.?[0-9]+)\.jpg$", fname, re.I)
    return float(m.group(1)) if m else None

# ────────────────────────────── EVALUATION ──────────────────────────
def evaluate_folder(folder: str, K: int = 20) -> Dict[str, float]:
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(("png", "jpg", "jpeg")) and
               "ess" not in f and "intermediate_rewards" not in f]
    if not imgs:
        raise RuntimeError("no images")

    qual = {k: [] for k in ["aesthetic", "hps", "imagereward", "pick", "clip"]}
    clip_e, dream_e, lpips_imgs = [], [], []

    # ─── 프롬프트별 임베딩을 따로 모으기 위한 dict ───
    clip_grp, dream_grp = {}, {}               # {prompt: [emb, ...]}
    
    for path in imgs:
        try:
            prompt = identify_prompt(os.path.basename(path))
        except ValueError:
            continue

        aest = parse_aesthetic(path)
        if aest is not None:
            qual["aesthetic"].append(aest)

        img_t = torchvision.transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            qual["clip"].append(clip_fn(img_t, prompt).item())
            qual["hps"].append(hps_fn(img_t, prompt).item())
            qual["imagereward"].append(imagereward_fn(img_t, prompt).item())
            qual["pick"].append(pick_fn(img_t, prompt).item())

            c_emb = clip_model.get_image_features(
                preprocess_clip(path).unsqueeze(0).to(device)).cpu().numpy().squeeze()
            d_emb = embed_dreamsim(path).cpu().numpy().squeeze()

            clip_e.append(c_emb)
            dream_e.append(d_emb)
            lpips_imgs.append(preprocess_lpips(path).to(device))

            # ─── prompt 그룹에 추가 ───
            clip_grp.setdefault(prompt, []).append(c_emb)
            dream_grp.setdefault(prompt, []).append(d_emb)

    clip_e, dream_e = map(np.asarray, (clip_e, dream_e))

    # ───────────── 전체(모든 이미지) 다양성 ─────────────
    clip_d   = pdist(clip_e,  "cosine")
    dream_d  = pdist(dream_e, "cosine")

    # ───────────── prompt-별 다양성 ─────────────
    clip_d_prompt = []
    for emb_list in clip_grp.values():
        if len(emb_list) > 1:                 # 한 prompt에 이미지 ≥2장일 때만 계산
            clip_d_prompt.append(pdist(np.asarray(emb_list), "cosine").mean())
    dream_d_prompt = []
    for emb_list in dream_grp.values():
        if len(emb_list) > 1:
            dream_d_prompt.append(pdist(np.asarray(emb_list), "cosine").mean())

    # ────────────────── 나머지 지표 ──────────────────
    cov     = np.cov(clip_e, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[-K:]
    tce     = (K/2)*np.log(2*np.pi*np.e) + 0.5*np.sum(np.log(eigvals))
    lp_d    = [lpips_model(lpips_imgs[i], lpips_imgs[j]).item()
               for i in range(len(lpips_imgs)) for j in range(i+1, len(lpips_imgs))]

    def ms(a):
        return (float(np.mean(a)), float(np.std(a))) if a else (float("nan"), float("nan"))

    res = {}
    for k in qual:
        res[f"mean_{k}"], res[f"std_{k}"] = ms(qual[k])

    # ─── 전체 다양성 ───
    res.update(
        clip_diversity=float(np.mean(clip_d)),
        clip_diversity_se=float(np.std(clip_d)/math.sqrt(clip_d.size)),
        dreamsim_diversity=float(np.mean(dream_d)),
        dreamsim_diversity_se=float(np.std(dream_d)/math.sqrt(dream_d.size)),
    )

    # ─── prompt-별 다양성 ───
    res.update(
        clip_diversity_prompt=float(np.mean(clip_d_prompt)) if clip_d_prompt else float("nan"),
        clip_diversity_prompt_se=float(np.std(clip_d_prompt)/math.sqrt(len(clip_d_prompt)))
                                 if clip_d_prompt else float("nan"),
        dreamsim_diversity_prompt=float(np.mean(dream_d_prompt)) if dream_d_prompt else float("nan"),
        dreamsim_diversity_prompt_se=float(np.std(dream_d_prompt)/math.sqrt(len(dream_d_prompt)))
                                     if dream_d_prompt else float("nan"),
    )

    # ─── 나머지 ───
    res.update(
        tce_clip=float(tce),
        lpips_mean=float(np.mean(lp_d)),
        lpips_std=float(np.std(lp_d)),
    )

    # 저장
    with open(os.path.join(folder, "eval_metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerow(res.keys())
        csv.writer(f).writerow([f"{v:.5f}" for v in res.values()])

    torch.cuda.empty_cache()
    return res

# ────────────────────────────── MAIN ────────────────────────────────
def collect_folders(base_dir: str, every: int):
    eval_f, search_f, eval_idx, search_idx = [], [], [], []

    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue

        # ─── eval_<epoch>-improve_1 ───
        m = re.match(r"eval_(\d+)-improve_1$", name)
        if m:
            epoch = int(m.group(1))
            if epoch == 1 or epoch % every == 0:       # ★ epoch 1 무조건 포함
                eval_f.append(name)
                eval_idx.append(epoch)
            continue

        # ─── search_<step> ───
        m = re.match(r"search_(\d+)$", name)
        if m:
            step = int(m.group(1))
            if step % every == 0:         # ★ step 1 무조건 포함 (원하면 유지)
                search_f.append(name)
                search_idx.append(step)

    # 정렬
    eval_f.sort(key=lambda x: int(re.match(r"eval_(\d+)", x).group(1)))
    search_f.sort(key=lambda x: int(re.match(r"search_(\d+)", x).group(1)))
    idx_sorted = sorted(set(eval_idx + search_idx))
    return eval_f, search_f, idx_sorted

def run(base_dir: str, every: int):
    out_dir = os.path.join(base_dir, "eval"); os.makedirs(out_dir, exist_ok=True)
    eval_f, search_f, idxs = collect_folders(base_dir, every)

    results = {}
    for fld in tqdm(eval_f + search_f, desc=f"Evaluating every {every} folders"):
        try:  results[fld] = evaluate_folder(os.path.join(base_dir, fld))
        except Exception as e: print(f"[!] {fld} skipped ({e})")

    # CSV
    csv_path = os.path.join(out_dir, "all_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        hdr = ["folder"] + list(next(iter(results.values())).keys())
        csv.writer(f).writerow(hdr)
        for fld in eval_f + search_f:
            if fld in results:
                csv.writer(f).writerow([fld] + [f"{results[fld][k]:.5f}" for k in hdr[1:]])
    print("✓ saved", csv_path)

    # plotting spec
    metrics = [
        ("mean_aesthetic","std_aesthetic","Aesthetic","aesthetic"),
        ("mean_hps","std_hps","HPS","hps"),
        ("mean_imagereward","std_imagereward","ImageReward","imagereward"),
        ("mean_pick","std_pick","Pick","pick"),
        ("mean_clip","std_clip","CLIP","clip"),
        ("clip_diversity","clip_diversity_se","CLIP-Div","clip_diversity"),
        ("dreamsim_diversity","dreamsim_diversity_se","DreamSim-Div","dreamsim_diversity"),
        ("tce_clip",None,"TCE","tce_clip"),
        ("lpips_mean","lpips_std","LPIPS-Mean","lpips_mean"),
        ("clip_diversity_prompt", "clip_diversity_prompt_se", "CLIP-Div (prompt)", "clip_div_prompt"),
        ("dreamsim_diversity_prompt", "dreamsim_diversity_prompt_se", "DreamSim-Div (prompt)", "dreamsim_div_prompt"),
        ("lpips_diversity_prompt","lpips_diversity_prompt_se","LPIPS-Div (prompt)","lpips_div_prompt"),
        ("tce_clip_prompt", "tce_clip_prompt_se", "TCE (prompt)", "tce_clip_prompt"),
    ]

    for key,std_key,label,suf in metrics:
        mean_eval   = [results.get(f"eval_{e}-improve_1",{}).get(key,np.nan)   for e in idxs]
        mean_search = [results.get(f"search_{e}",{}).get(key,np.nan)                 for e in idxs]
        std_eval    = [results.get(f"eval_{e}-improve_1",{}).get(std_key,0.0)  for e in idxs] if std_key else [0.0]*len(idxs)
        std_search  = [results.get(f"search_{e}",{}).get(std_key,0.0)                for e in idxs] if std_key else [0.0]*len(idxs)

        plt.figure(figsize=(9,5))
        plt.plot(idxs,mean_eval,'o-',label=f"{label} (eval)")
        plt.fill_between(idxs, np.array(mean_eval)-np.array(std_eval), np.array(mean_eval)+np.array(std_eval), alpha=.2)
        plt.plot(idxs,mean_search,'x--',label=f"{label} (search)")
        plt.fill_between(idxs, np.array(mean_search)-np.array(std_search), np.array(mean_search)+np.array(std_search), alpha=.2)

        plt.xlabel(f"Epoch/step (every {every})"); plt.ylabel(label)
        plt.title(f"{label}: eval vs search"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        out_png = os.path.join(out_dir,f"metrics_eval_vs_search_{suf}.png")
        plt.savefig(out_png,dpi=200); plt.close(); print("✓",out_png)

# ────────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Evaluate eval/search folders every N and plot metrics.")
    ap.add_argument("--base_dir", required=True, help="Directory containing eval_epoch-*/search_* subfolders")
    ap.add_argument("--every", type=int, default=3, help="Stride for epochs/steps (default 3)")
    args = ap.parse_args()
    run(args.base_dir, args.every)