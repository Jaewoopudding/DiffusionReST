# 2025.03.21 aesthetic 기본 베이스라인 배치사이즈 64
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch scripts/train.py --config config/vanila_aesthetic_64.py --config.run_name vanila 
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch scripts/train.py --config config/vanila_compressibility_64.py --config.run_name vanila 
