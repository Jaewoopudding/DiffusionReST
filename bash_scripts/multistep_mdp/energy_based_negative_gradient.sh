
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 29503 scripts/train_mcts_mle.py \
--config config/svdd_aesthetic_mle.py \
--config.num_epochs 30 \
--config.search.duplicate 2 \
--config.run_name search_grad \
--config.train.learning_rate 1e-3 \
--config.train.total_batch_size 2 \
--config.train.gradient_steps_per_improve_step 500 \
--config.train.improve_steps 1 \
--config.sample.num_batches_per_epoch 32 \
--config.prompt_fn simple_animals \
--config.train.kl_coef 0.00 \
--config.train.type 'energy_based_negative_gradient' \
--config.eval.num_images_per_prompt 1 \
--config.sample.num_prompts_per_batch 32 \
--config.search.value_gradient \
--config.reward_fn aesthetic_score_diff \
--config.search.kl_lagrangian_coef 0.005 \