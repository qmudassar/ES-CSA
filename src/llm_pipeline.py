# Description: Module for complete HuggingFace Text Generation LLM pipeline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initializing Pipeline.

def create_pipeline(model=None, tokenizer=None, model_name=None,
                    max_new_tokens=128, temperature=0.3,
                    top_k=20, top_p=0.7):
    """
    Create and return a HuggingFace text-generation pipeline.
    """
    if model is None or tokenizer is None:
        assert model_name is not None, "You must provide either model/tokenizer or model_name"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    llm_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return llm_pipeline