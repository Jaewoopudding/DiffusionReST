CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/amortize_das.py \
--config.data.reward_filtering_percentile 50 \
--config.train.sft_negative_gradient \