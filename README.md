# Diffusion Alignment as Variational Expectation-Maximization (Text-to-Image Generation)

This repository is the official codebase of [Diffusion Alignment as Variational Expectation-Maximization (DAV)](https://arxiv.org/abs/2510.00502) for Text-to-Image generation. This implementation optimizes Stable Diffusion v1.5 based on aesthetic score, compressibility, and incompressibility rewards. If you are interested in applying DAV to DNA sequence optimization, please refer to [here](https://your-link-here.com).

<p align="center">
  <img src="assets/t2i_comparison.png" width="100%">
</p>

## Installation

```bash
conda env create -f environment.yml
conda activate dav-t2i
pip install -e .
```

## Usage

Run an experiment using one of the provided bash scripts:

```bash
bash bash_scripts/aesthetic.sh
```

Or launch directly with custom flags:

```bash
accelerate launch scripts/dav.py \
  --config config/dav.py \
  --config.reward_fn aesthetic_score_diff \
  --config.search.duplicate 4 \
  --config.train.train_kl 0.01 \
  --config.seed 0
```

All hyperparameters have defaults in `config/dav.py` and can be overridden via command-line flags.

## Hyperparameters

### Search

| Flag | Description |
|------|-------------|
| `search.duplicate` | Search width: number of particle at each timestep. |
| `search.search_kl` | Temperature for importance sampling weights and value-gradient scaling. Lower values make search more exploitative. |
| `search.value_gradient` | Use reward gradient to guide the denoising policy. Must be set to False if the reward function is non-differentiable. |
| `search.gamma` | Discount factor applied to value-gradient magnitude across timesteps. |
| `search.importance_sampling` | Use importance-weighted softmax for node selection instead of uniform. |
| `search.hill_climbing` | If True, search uses the current (training) model; if False, uses the frozen pretrained model. |

### Training

| Flag | Description |
|------|-------------|
| `train.train_kl` | KL regularization coefficient. When > 0, adds `train_kl * MSE(noise_pred, ref_noise_pred)` to the MLE loss to prevent overoptimization. |
| `train.improve_steps` | Number of training iterations over collected samples per epoch. |
| `train.learning_rate` | AdamW learning rate. |
| `train.timestep_fraction` | Fraction of trajectory timesteps used for training (1.0 = all). |
| `train.accumulation_multipler` | Multiplier for gradient accumulation steps beyond the default per-timestep accumulation. |
| `train.max_grad_norm` | Max gradient norm for clipping. |

### Sampling

| Flag | Description |
|------|-------------|
| `sample.num_batches_per_epoch` | Number of sampling batches per epoch. Total samples per epoch = `batch_size * num_batches_per_epoch * num_gpus`. |
| `sample.num_prompts_per_batch` | Number of distinct prompts sampled per epoch. Images per prompt = `num_batches_per_epoch / num_prompts_per_batch`. |
| `sample.num_steps` | Number of DDIM denoising steps. |
| `sample.eta` | DDIM stochasticity (must be > 0 for search to produce diverse candidates). |
| `sample.guidance_scale` | Classifier-free guidance scale. |

### General

| Flag | Description |
|------|-------------|
| `num_epochs` | Total training epochs. |
| `save_freq` | Checkpoint saving frequency (epochs). |
| `eval.eval_freq` | Evaluation frequency (epochs). |
| `initial_search` | If True, initial latent shape is scaled by `duplicate`. |


## Acknowledgement

This codebase builds upon the [DDPO-pytorch](https://github.com/kvablack/ddpo-pytorch) implementation. We thank the authors for making their code publicly available.