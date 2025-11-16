#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import get_context, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

load_dotenv()

def main():
    print(f"Running test...")
    parser = argparse.ArgumentParser(
        description="Test on Verilog-eval benchmark using various AI models (multi-GPU + batching)."
    )
    args = parser.parse_args()

if __name__=="__main__":
    main()