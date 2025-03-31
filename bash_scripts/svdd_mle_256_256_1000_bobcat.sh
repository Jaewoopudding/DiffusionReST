#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:2
#SBATCH -J jaewoo
#SBATCH -o /home/mila/y/yunt/jaewoo/DiffusionReST/slurm_logs/%x-%j.out

# cd /home/mila/y/yunt/jaewoo/DiffusionReST
# export PYTHONPATH=/home/mila/y/yunt/jaewoo/DiffusionReST:$PYTHONPATH
# export HF_HOME="/network/scratch/y/yunt/huggingface/datasets"

# module --quiet purge
# module --quiet load anaconda/3
# module --quiet load cuda/12.6.0/cudnn/9.3
# conda activate ddpo
# wandb login --relogin 4976cca9d8aba7c6e3ab132426742addc6ddedd3
# # huggingface-cli login --token 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

accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py --config config/svdd_aesthetic_mle.py --config.run_name svdd_mle_512_256_1000_mean_filter_1e5 --config.train.learning_rate 1e-5 --config.train.total_batch_size 64 --config.train.gradient_steps_per_improve_step 500 --config.sample.num_batches_per_epoch 16
# CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port 29501 scripts/train_mcts_mle.py --config config/svdd_compressibility_64.py --config.run_name SVDD+PPO_comp 
# CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py --config config/svdd_aesthetic_mle_debug.py --config.run_name SVDD+MLE