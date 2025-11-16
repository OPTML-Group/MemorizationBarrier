import torch
import torch.nn as nn
import json
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

#Define IB loss
def IB_loss(logits, labels, attention_mask=None, label_smoothing=0.0, ignore_index=-100):

    """
    Information Bottleneck loss with label smoothing extension.
    
    Args:
        logits: (batch, seq, vocab)
        labels: (batch, seq)
        label_smoothing: float in [0, 1]. 0 = normal CE.
        ignore_index: label to ignore (usually -100).
    """
 
    shift_logits = logits[:, :-1, :].contiguous()   # (batch, seq-1, vocab)
    shift_labels = labels[:, 1:]       # (batch, seq-1)

 
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:]
    else:
        shift_mask = None

    valid_label_mask = (shift_labels != ignore_index)
    valid_mask = valid_label_mask & shift_mask.bool() if shift_mask is not None else valid_label_mask

    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    shift_labels = shift_labels.reshape(-1)
    valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=True)[0]

    shift_logits = shift_logits[valid_indices]
    shift_labels = shift_labels[valid_indices]

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
 
    loss = loss_fct(shift_logits, shift_labels)
    return loss

## Vaiational Adapter
class VatiationalAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim=256):
        super().__init__()
        self.mu = nn.Linear(hidden_size, bottleneck_dim)
        self.logvar = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)

    def forward(self, h):
        mu, logvar = self.mu(h), self.logvar(h)
        logvar = torch.clamp(logvar, min=-4.0, max=4.0)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        hbar = self.up(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return hbar, kl
    
##IB model
class ModelWithIB(nn.Module):
    def __init__(self,
                 base_id,
                 target_layer=20,
                 bottleneck_dim=256,
                 beta=1e-2,
                 alpha=0.2,
                 Lora_config_path=None):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        #load LoRA
        if Lora_config_path is not None:
            with open(Lora_config_path, "r") as f:
                config = json.load(f)

            config_obj = LoraConfig(**config)
            base_model = get_peft_model(base_model, config_obj)
        
        self.lm = base_model
        self.target_layer = target_layer
        H = self.lm.config.hidden_size
        self.ib = VatiationalAdapter(H, bottleneck_dim=bottleneck_dim)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.lm(input_ids,
                      attention_mask=attention_mask,
                      output_hidden_states=True,
                      return_dict=True,)
        
        h = out.hidden_states[self.target_layer]
        h_bar, kl = self.ib(h)

        logits_ib = self.lm.lm_head(h_bar)
        logits_ce = out.logits

        ce_ib = IB_loss(logits=logits_ib, labels=labels)
        ce_orig = IB_loss(logits=logits_ce, labels=labels)
        loss_ib = self.alpha *(kl + self.beta * ce_ib)
        loss = ce_orig + loss_ib

        return {
            "loss": loss,
            "logits": logits_ib,
            "loss_ib": loss_ib.detach(),
            "ce_ib": ce_ib.detach(),
            "ce_orig": ce_orig.detach(),
            "kl_loss": kl.detach(),
        }



        
            
        

