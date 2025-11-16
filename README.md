# Breaking Memorization Barriers in LLM Code Fine-Tuning via Information Bottleneck for Improved Generalization
## Overview
**Memorization barrier:**  : The phenomenon where *LLM code fine-tuning* starts from a base model $\theta_0$ that already strongly memorizes the fine-tuning set $\mathcal{D}_{\text{code}}$, placing optimization in a state that the conventional fine-tuning objective struggles to escape — thereby leading to poor generalization on downstream code tasks.



## Training IB-FT
```bash
export WANDB_API_KEY=XXX
export HUGGINGFACE_TOKEN=XXX

torchrun --nproc_per_node 1  train_ib.py \
    --base_model="deepseek-ai/deepseek-coder-7b-instruct-v1.5"\
    --training_config="./config/training_args.json"\
    --adapter_config="./config/adapter_config.json"\
    --dataset="henryen/origen_dataset_instruction"\
    --use_deepspeed
```
