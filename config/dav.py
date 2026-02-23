import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 0
    config.logdir = "logs"
    config.num_epochs = 100
    config.save_freq = 5
    config.num_checkpoint_limit = 50
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.use_lora = True
    config.initial_search = False

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 50
    sample.eta = 1.0
    sample.guidance_scale = 5.0
    sample.batch_size = 1
    sample.num_batches_per_epoch = 8
    sample.num_prompts_per_batch = 8

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.learning_rate = 1e-3
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.max_grad_norm = 1.0
    train.improve_steps = 1
    train.timestep_fraction = 1.0
    train.train_kl = 0.01
    train.accumulation_multipler = 1

    ###### Evaluation ######
    config.eval = eval = ml_collections.ConfigDict()
    eval.eval_freq = 10

    ###### Prompt Function ######
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}

    ###### Reward ######
    config.reward_fn = "aesthetic_score_diff"

    ###### Searching ######
    config.search = search = ml_collections.ConfigDict()
    search.duplicate = 4
    search.value_gradient = True
    search.search_kl = 0.005
    search.importance_sampling = True
    search.gamma = 0.90
    search.hill_climbing = True

    return config
