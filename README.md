<div align='center'>
 
# Breaking Memorization Barriers in LLM Code Fine-Tuning via Information Bottleneck for Improved Generalization

[![Intel](https://img.shields.io/badge/Intel-Collaboration-0071C5?logo=intel&logoColor=white)]()
[![preprint](https://img.shields.io/badge/arXiv-2510.16022-B31B1B)](https://arxiv.org/abs/2510.16022)
[![collection](https://img.shields.io/badge/HuggingFace-Collection_(TBD)-yellow)](https://huggingface.co/collections/OPTML-Group/TBD)
[![issues](https://img.shields.io/badge/Issues-Welcome!-yellow)](https://github.com/OPTML-Group/MemorizationBarrier/issues)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/MemorizationBarrier?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/MemorizationBarrier)](https://github.com/OPTML-Group/MemorizationBarrier)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/MemorizationBarrier)](https://github.com/OPTML-Group/MemorizationBarrier)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/MemorizationBarrier)](https://github.com/OPTML-Group/MemorizationBarrier)

</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;">
        <strong style="font-size: 18px;">Figure 1:</strong>
        Overview and key results of our Information Bottleneck–based code fine-tuning framework.
      </em>
    </td>
  </tr>
</table>

This is the official code repository for the paper
[Breaking Memorization Barriers in LLM Code Fine-Tuning via Information Bottleneck for Improved Generalization](https://arxiv.org/abs/2510.16022).















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
