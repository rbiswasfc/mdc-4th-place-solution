# Make Data Count - Finding Data References

This repo contains the solution code for the [Make Data Count - Finding Data References](https://www.kaggle.com/competitions/make-data-count-finding-data-references) Kaggle competition, which won the 4th place. The full solution is described [here](https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/4th-place-solution). Please refer to the following sections for details on dependencies, training, and synthetic data labelling agent. If you have any questions or ran into any issues, please feel free to open an issue.

# 1 Setup
## 1.1 Compute
- 8x H100 SMX 80GB HBM3
- Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz (32 vCPUs)
- RAM: 256 GB
- Disk space: 1 TB

## 1.2 Environment

Please clone the repo and install the dependencies by running the following commands:

```bash
conda create -n mdc python=3.12 pandas matplotlib numpy ipykernel jupyter
conda activate mdc
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # 2.7.0
pip install -r requirements.txt
python -m ipykernel install --user --name mdc --display-name "mdc"
pip install flash-attn --no-build-isolation # 2.8.1
pip install deepspeed # 0.17.2
pip install "huggingface_hub[hf_transfer]"
```

## 1.3 Download Datasets
Please export your Kaggle username and token to the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`. It will be needed the download the competition datasets. The API keys can be obtained from the [Kaggle Settings page](https://www.kaggle.com/settings).

```
export KAGGLE_USERNAME=******
export KAGGLE_KEY=******
```

Next, download the required datasets by running:

```
python download_datasets.py
```

## 1.4 Download Models
Please download the models from HF Hub:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-14B
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-72B
```

## 1.5 Accelerate Setup
Models were trained using the HF accelerate library with DDP. Specifically, the following accelerate config was used:

```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```


# 2 Training
## 2.1 Type Classifier

If you want to track training runs using wandb, please log in to your wandb account by running wandb login from the terminal. Otherwise, you can set `use_wandb=false` in the following commands.

```bash
accelerate launch ./code/train_classifier.py --config-name conf_14b_warmup use_wandb=true
accelerate launch ./code/train_classifier.py --config-name conf_14b_continue use_wandb=true full_fit=true

accelerate launch ./code/train_classifier.py --config-name conf_32b_warmup use_wandb=true
accelerate launch ./code/train_classifier.py --config-name conf_32b_continue use_wandb=true full_fit=true

accelerate launch ./code/train_classifier.py --config-name conf_72b use_wandb=true
```

## 2.2 Candidate Filter

```bash
accelerate launch ./code/train_filter.py --config-name conf_pl use_wandb=true full_fit=true
```

# 3 Synthetic Data Labelling Agent

Our synthetic dataset used for LLM warmup can be found [here](https://www.kaggle.com/datasets/conjuring92/mdc-type-synthetic-v1).

It is created using a tool calling agent demoed [here](https://www.kaggle.com/code/conjuring92/mdc-tool-calling-agent-demo). To generate more synthetic data, please first export the OpenAI API key to the environment variable:
Please first export the OpenAI API key to the environment variable:

```
export OPENAI_API_KEY=******
```

Then, run the following command:

```bash
python code/labelling_agent.py
```


# 4 Inference

My best selected inference notebook can be found [here](https://www.kaggle.com/code/conjuring92/mdc-a12-mdc-pipeline?scriptVersionId=260735888).

