# 2025.03.21 aesthetic 기본 베이스라인 배치사이즈 64
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29501 scripts/train_mcts.py --config config/svdd_aesthetic_64.py --config.run_name SVDD+PPO 
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --main_process_port 29501 scripts/train_mcts.py --config config/svdd_compressibility_64.py --config.run_name SVDD+PPO_comp 
