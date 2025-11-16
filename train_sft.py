import os
import argparse
import json
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers.integrations import TensorBoardCallback
import wandb
from transformers.trainer_utils import is_main_process
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.ModelWithIB import ModelWithIB
from util.util import get_training_args,SaveLoRAEverySave, WandbLoggingCallback
from util.process_data import format_origen


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True, help="Base model")
parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--adapter_config", type=str, required=True, help="Path to adapter configuration file")
parser.add_argument("--training_config", type=str, required=True, help="Path to training configuration file")
parser.add_argument("--use_deepspeed", action="store_true")
parser.add_argument("--target_layer", type=int, default=20)
parser.add_argument("--num_proc", type=int, default=8)
parser.add_argument('--max_length', type=int, default=2048)
args = parser.parse_args()

def main():
    print(f"Training...{args.base_model} on {args.dataset}")

    set_seed(args.seed)
    training_args = get_training_args(args.training_config)

    if args.use_deepspeed:
        training_args.deepspeed = "config/ds_config.json"
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    if args.adapter_config is not None:
        with open(args.adapter_config, "r") as f:
            config = json.load(f)

        config_obj = LoraConfig(**config)
        model = get_peft_model(model, config_obj)

    if args.dataset == "henryen/origen_dataset_instruction":
        print(f"Dealing with dataset {args.dataset}")
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.map(format_origen, remove_columns=dataset.column_names)
        def tokenize_origen(example):
            prompt = example["Instruction"] + "\n"
            response = example["Response"]
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            input_ids = prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + response_ids

            max_len = args.max_length
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            padding_length = max_len - len(input_ids)

            if padding_length > 0:
                # pad to max_len
                input_ids += [pad_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length
            else:
                # if over max_len and cut
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]

            return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        tokenized_dataset = dataset.map(
        tokenize_origen, 
        num_proc=min(args.num_proc, os.cpu_count()),  # parallel workers (processes)
        remove_columns=dataset.column_names,
        load_from_cache_file=True,)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[TensorBoardCallback(), WandbLoggingCallback(),SaveLoRAEverySave(keep_last_n = training_args.save_total_limit)],
              )
        
        trainer.train()
        model.lm.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        if is_main_process(training_args.local_rank):
            wandb.finish()
        

if __name__=="__main__":
    main()