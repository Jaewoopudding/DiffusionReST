# DiffExIT

## Installation

```bash
pip install -e. 
conda env create -f environment.yml
conda activate DiffExIT
```

## Usage
```bash
bash bash_scripts/minsu/minsu_poc_1.sh
```

## Important Hyperparameters


--config.search.duplicate : search width for tree search

--config.train.total_batch_size : train batch size regardless of gpu num

--config.train.gradient_steps_per_improve_step : gradient steps per one improvement step

--config.train.improve_steps : number of improvement steps

--config.sample.num_batches_per_epoch : number of images per one grow step per single gpu, total generation count is gpu_count * num_batches_per_epoch

--config.train.kl_coef : If it is set as non-zero, kl regularization for sft will be activated

--config.train.type : dpo or sft

--config.sample.num_prompts_per_batch : the number of prompts per one grow step. the number of image of single prompt is gpu_count * num_batches_per_epoch / num_prompts_per_batch

