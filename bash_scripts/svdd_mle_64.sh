# 2025.03.21 aesthetic 기본 베이스라인 배치사이즈 64
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py --config config/svdd_aesthetic_mle_64.py --config.run_name SVDD+MLE1
# CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port 29501 scripts/train_mcts_mle.py --config config/svdd_compressibility_64.py --config.run_name SVDD+PPO_comp 
# CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py --config config/svdd_aesthetic_mle_debug.py --config.run_name SVDD+MLE