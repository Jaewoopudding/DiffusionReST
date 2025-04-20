wandb init --entity gda-for-orl --project ddpo-pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 29502 scripts/train_mcts_mle.py \
--config config/svdd_aesthetic_mle.py \
--config.search.duplicate 70 \
--config.run_name simple_animals_128/64-500*8_num_prompts_per_batch8_60 \
--config.train.learning_rate 1e-8 \
--config.train.total_batch_size 64 \
--config.train.gradient_steps_per_improve_step 250 \
--config.train.improve_steps 10 \
--config.sample.num_batches_per_epoch 32 \
--config.prompt_fn simple_animals \
--config.train.kl_coef 0.1 \
--config.train.type 'dpo' \
--config.eval.num_images_per_prompt 8 \
--config.sample.num_prompts_per_batch 8



## SCRIPT FOR 8 GPU ENV ## 