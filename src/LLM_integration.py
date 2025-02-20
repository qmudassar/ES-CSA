import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phi-2 LLM Integration

model_name = "microsoft/phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print("Tokenizer loaded successfully.")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto"
).to(device)

print(f"Model Loaded Successfully on {device}!")


def generate_response(prompt):
    print(f"Received prompt: {prompt}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    print("Generation completed!")
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test Query

query = "What are the benefits of Phi-2 for chatbots?"
response = generate_response(query)
print(f"Response: {response}")