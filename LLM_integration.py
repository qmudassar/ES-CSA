import transformers
import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

## Mistral-7B-Instruct-v0.1

load_dotenv()

model_name = os.getenv("MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch_dtype, device_map="auto"
).to(device)

print("Model Loaded Successfully!")


## Test Query Function

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

## Test Query Example

query = "What is my remaining data browsing allowance?"
response = generate_response(query)
print("Response:", response)