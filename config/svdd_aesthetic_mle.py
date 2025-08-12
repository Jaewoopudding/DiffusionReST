import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 5
    # number of epochs between saving model checkpoints.
    config.save_freq = 5
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 50
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    config.initial_search = False

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 256
    sample.num_prompts_per_batch = 4 # for prompt-conditional buffer

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 1e-5
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.improve_steps = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the PPO clip range.
    train.clip_range = 1e-4
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    
    
    # number of gradient steps to take per improve step
    train.gradient_steps_per_improve_step = 1000
    # number of total batch size used at improve step
    train.total_batch_size = 256
    # kl regularizer coefficient
    train.kl_coef = 0.00
    # DPO or SFT?
    train.type = 'sft' # dpo or sft
    train.beta_dpo = 5000
    train.negative_gradient = True
    train.accumulation_multipler = 1
    train.kl_lagrangian_coef = 0.
    
    config.eval = eval = ml_collections.ConfigDict()
    eval.num_images_per_prompt = 8
    # frequency of evaluation (every N epochs)
    eval.eval_freq = 10

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    # config.prompt_fn = "imagenet_animals"
    config.prompt_fn = "imagenet_animals_all" # for prompt-conditional buffer
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = "aesthetic_score" # aesthetic_score jpeg_compressibility
    config.eval_fn = "multi_reward_evaluation"
    config.multistep_mdp = True

    ###### Per-Prompt Stat Tracking ######
    # when enabled, the model will track the mean and std of reward on a per-prompt basis and use that to compute
    # advantages. set `config.per_prompt_stat_tracking` to None to disable per-prompt stat tracking, in which case
    # advantages will be calculated using the mean and std of the entire batch.
    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    # number of reward values to store in the buffer for each prompt. the buffer persists across epochs.
    config.per_prompt_stat_tracking.buffer_size = 16
    # the minimum number of reward values to store in the buffer before using the per-prompt mean and std. if the buffer
    # contains fewer than `min_count` values, the mean and std of the entire batch will be used instead.
    config.per_prompt_stat_tracking.min_count = 16
    
    ###### Searching ######
    config.search = search = ml_collections.ConfigDict()
    
    search.nfe_per_action = 1
    search.duplicate = 10
    search.expansion_coef = 0.0
    search.progressive_widening = False
    search.pw_alpha = 0.0
    search.value_gradient = False
    search.kl_lagrangian_coef = 0.005
    search.tempering_gamma = 0.008
    search.jump_policy = False
    search.importance_sampling = True
    search.gamma = 0.90
    search.hill_climbing = True


    config.buffer = buffer = ml_collections.ConfigDict()
    
    buffer.per_prompt_filtering_flag = True
    buffer.per_prompt_select_flag = False
    buffer.reward_filtering_criteria = 0.0
    buffer.clip_score_filtering_criteria = 0.0
    buffer.off_policy_subset_size = 0
    
    return config