import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Base path
images_root = "/home/jaewoo/DiffusionReST/images"

# Output directories
os.makedirs("plots/search", exist_ok=True)
os.makedirs("plots/eval", exist_ok=True)
os.makedirs("plots/search_by_epoch", exist_ok=True)
os.makedirs("plots/eval_by_epoch", exist_ok=True)

# All metrics to compare against aesthetic
y_metrics = [
    'mean_hps', 'mean_imagereward', 'mean_pick', 'mean_clip',
    'clip_diversity', 'dreamsim_diversity', 'tce_clip', 'lpips_mean'
]

# CSV file paths
all_csv_paths = glob.glob(os.path.join(images_root, "**", "eval", "all_metrics.csv"), recursive=True)

for data_type in ['search_', 'eval_']:
    out_dir_reward = f"plots/{data_type.strip('_')}"
    out_dir_epoch = f"plots/{data_type.strip('_')}_by_epoch"

    # ---- Plot 1: Aesthetic vs each metric ----
    for y_metric in y_metrics:
        plt.figure(figsize=(8, 6))
        for csv_path in all_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                df = df[df["folder"].str.startswith(data_type)]
                if df.empty:
                    continue

                experiment_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
                df["step"] = df["folder"].str.extract(rf'{data_type}(\d+)').astype(float)
                df = df.sort_values("step")

                x = df["mean_aesthetic"]
                y = df[y_metric]

                plt.plot(x, y, marker='o', label=experiment_name, alpha=0.7)
            except Exception as e:
                print(f"[aesthetic-x] Error in {csv_path}: {e}")

        plt.xlabel("Mean Aesthetic Score")
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f"{data_type.strip('_').capitalize()} - Aesthetic vs {y_metric}")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_reward, f"aesthetic_vs_{y_metric}.png"))
        plt.close()

    # ---- Plot 2: Epoch vs each metric ----
    for y_metric in y_metrics + ["mean_aesthetic"]:
        plt.figure(figsize=(8, 6))
        for csv_path in all_csv_paths:
            try:
                df = pd.read_csv(csv_path)
                df = df[df["folder"].str.startswith(data_type)]
                if df.empty:
                    continue

                experiment_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
                df["step"] = df["folder"].str.extract(rf'{data_type}(\d+)').astype(float)
                df = df.sort_values("step")

                x = df["step"]
                y = df[y_metric]

                plt.plot(x, y, marker='o', label=experiment_name, alpha=0.7)
            except Exception as e:
                print(f"[epoch-x] Error in {csv_path}: {e}")

        plt.xlabel("Epoch")
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.title(f"{data_type.strip('_').capitalize()} - Epoch vs {y_metric}")
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_epoch, f"epoch_vs_{y_metric}.png"))
        plt.close()
