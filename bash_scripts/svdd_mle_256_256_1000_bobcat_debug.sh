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


CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py \
--config config/svdd_aesthetic_mle.py \
--config.search.duplicate 1 \
--config.run_name simple_animals_128/53-500*6 \
--config.train.learning_rate 1e-5 \
--config.train.total_batch_size 1 \
--config.train.gradient_steps_per_improve_step 4 \
--config.train.improve_steps 2 \
--config.sample.num_batches_per_epoch 2 \
--config.prompt_fn simple_animals \
--config.train.kl_coef 0.1 \
--config.train.type 'dpo' \
--config.eval.num_images_per_prompt 1 \
--config.sample.num_prompts_per_batch 1

# CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port 29501 scripts/train_mcts_mle.py --config config/svdd_compressibility_64.py --config.run_name SVDD+PPO_comp 
# CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py --config config/svdd_aesthetic_mle_debug.py --config.run_name SVDD+MLEㅊs