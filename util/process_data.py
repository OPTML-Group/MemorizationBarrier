import os
import sys
import numpy as np
import torch


def format_origen(example):
    instruction = example.get("Instruction", "")
    response = example.get("Response", "")
    return {"Instruction": instruction, "Response": response}