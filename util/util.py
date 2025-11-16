import torch
import torch.nn as nn
import json
import numpy as np
import os
import wandb
import shutil
import glob

from transformers import (
    TrainingArguments,
    TrainerCallback
)

# ========= Load Training Args =========
def get_training_args(config_path: str) -> TrainingArguments:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg.pop("evaluation_strategy", None)
    return TrainingArguments(**cfg)



class SaveLoRAEverySave(TrainerCallback):
    """
    Save LoRA/adapter weights when the Trainer triggers a save event,
    and automatically prune older ones (keep last N).
    """
    def __init__(self, keep_last_n=3):
        super().__init__()
        self.keep_last_n = keep_last_n

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        # save using main process
        if not state.is_world_process_zero:
            return

        # Compatible model ：PeftModel / ModelWithIB(lm=PeftModel)
        peft_model = getattr(model, "lm", model)
        if not hasattr(peft_model, "save_pretrained"):
            return

        out = os.path.join(args.output_dir, f"adapter_step{state.global_step}")
        os.makedirs(out, exist_ok=True)
        peft_model.save_pretrained(out)   # save adapter_model.safetensors + adapter_config.json
        print(f"[LoRA] saved adapter to: {out}")

        #---cleanup old adapter directories---
        all_adapters = sorted(glob.glob(os.path.join(args.output_dir, "adapter_step*")))
        if len(all_adapters) > self.keep_last_n:
            for old_adapter in all_adapters[:-self.keep_last_n]:
                print(f"[Cleanip] Removing old adapters: {old_adapter}")
                shutil.rmtree(old_adapter, ignore_errors=True)


# ========= W&B Logging Callback =========
class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and (args.local_rank == -1 or args.local_rank == 0):
            flat_logs = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in logs.items()}
            wandb.log(flat_logs, step=state.global_step)