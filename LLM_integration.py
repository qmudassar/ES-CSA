import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

## Phi-2 Debugging Version

load_dotenv()

model_name = os.getenv("MODEL", "microsoft/phi-2")
device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")  # Debugging print

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print("Tokenizer loaded successfully.")  # Debugging print

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch_dtype, device_map="auto"
).to(device)

print(f"Model Loaded Successfully on {device}!")  # Debugging print

## Test Function
def generate_response(prompt):
    print(f"Received prompt: {prompt}")  # Debugging print
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print("Generating response...")  # Debugging print
    output = model.generate(input_ids, max_length=150)

    print("Generation completed!")  # Debugging print
    return tokenizer.decode(output[0], skip_special_tokens=True)

## Test Query
query = "What are the benefits of Phi-2 for chatbots?"
response = generate_response(query)
print("Response:", response)