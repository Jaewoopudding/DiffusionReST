import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
#
# 기본 images_root 경로를 현 프로젝트의 `images_to_plot` 폴더로 두고,
# 필요하면 CLI 인자로 덮어쓸 수 있도록 argparse를 사용한다.
# 예) python vs.py --images_root /path/to/other/images
parser = argparse.ArgumentParser(
    description="각 experiment 의 all_metrics.csv 를 읽어서 다양한 지표를 플롯합니다."
)
parser.add_argument(
    "--images_root",
    type=str,
    default="./images_to_plot",
    help="실험 결과가 저장된 루트 디렉터리 (experiment 하위에 eval/all_metrics.csv 가 존재해야 함)"
)
args = parser.parse_args()

images_root = args.images_root

# 결과 폴더
for d in [
    "plots/search", "plots/eval",
    "plots/search_by_epoch", "plots/eval_by_epoch"
]:
    os.makedirs(d, exist_ok=True)

# y축에 쓸 모든 지표
y_metrics = [
    # ─── 품질 지표 ───
    'mean_hps', 'mean_imagereward', 'mean_pick', 'mean_clip',
    # ─── 전체 다양성 ───
    'clip_diversity', 'dreamsim_diversity', 'tce_clip', 'lpips_mean',
    # ─── prompt-별 다양성 (추가) ───
    'clip_diversity_prompt', 'dreamsim_diversity_prompt',
    'tce_clip_prompt', 'lpips_diversity_prompt',
]

# eval/all_metrics.csv 경로 모두 수집
all_csv_paths = glob.glob(
    os.path.join(images_root, "**", "eval", "all_metrics.csv"), recursive=True
)

# ------------------------------------------------------------------
# 1. search_*  vs  eval_* 각각 플롯
# ------------------------------------------------------------------
for data_type in ['search_', 'eval_']:

    # 출력 디렉터리
    out_dir_reward = f"plots/{data_type.strip('_')}"
    out_dir_epoch  = f"plots/{data_type.strip('_')}_by_epoch"

    # ────────────────────────────────────────────────────────────
    # 1-1) Aesthetic - vs - 각 지표
    # ────────────────────────────────────────────────────────────
    for y_metric in y_metrics:
        plt.figure(figsize=(10, 5))
        for csv_path in all_csv_paths:
            try:
                df = pd.read_csv(csv_path)

                # search_* 또는 eval_* 행만
                df = df[df["folder"].str.startswith(data_type)]
                if df.empty:
                    continue

                # 실험 이름 (상위 폴더)
                experiment_name = os.path.basename(
                    os.path.dirname(os.path.dirname(csv_path))
                )

                # step/epoch 숫자 추출 & 정렬
                df["step"] = (
                    df["folder"].str.extract(rf'{data_type}(\d+)').astype(float)
                )
                df = df.sort_values("step")

                # 플롯
                plt.plot(
                    df["mean_aesthetic"], df[y_metric],
                    marker='o', label=experiment_name, alpha=0.7
                )
            except Exception as e:
                print(f"[aesthetic-x] {csv_path}: {e}")

        plt.xlabel("Mean Aesthetic Score")
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f"{data_type.strip('_').capitalize()} - Aesthetic vs {y_metric}")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_reward, f"aesthetic_vs_{y_metric}.png"))
        plt.close()

    # ────────────────────────────────────────────────────────────
    # 1-2) Epoch/Step - vs - 각 지표 (aesthetic 포함)
    # ────────────────────────────────────────────────────────────
    for y_metric in y_metrics + ["mean_aesthetic"]:
        plt.figure(figsize=(10, 5))
        for csv_path in all_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                df = df[df["folder"].str.startswith(data_type)]
                if df.empty:
                    continue

                experiment_name = os.path.basename(
                    os.path.dirname(os.path.dirname(csv_path))
                )

                df["step"] = (
                    df["folder"].str.extract(rf'{data_type}(\d+)').astype(float)
                )
                df = df.sort_values("step")

                plt.plot(
                    df["step"], df[y_metric],
                    marker='o', label=experiment_name, alpha=0.7
                )
            except Exception as e:
                print(f"[epoch-x] {csv_path}: {e}")

        plt.xlabel("Epoch" if data_type == 'eval_' else "Step")
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f"{data_type.strip('_').capitalize()} - Epoch vs {y_metric}")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_epoch, f"epoch_vs_{y_metric}.png"))
        plt.close()
