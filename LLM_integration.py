import pandas as pd
import numpy as np
import transformers
import torch
import os


## Downloading Mistral-7B-Instruct-v0.1

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
).to(device)

print(" Mistral-7B-Instruct-v0.1 Model Loaded Successfully!")