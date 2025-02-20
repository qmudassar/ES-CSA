# LangChain Integration w/ Phi-2 LLM

from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel, Field

class Phi2LLM(LLM, BaseModel):
    model_name: str = "microsoft/phi-2"
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: AutoTokenizer = Field(default=None, exclude=True)
    model: AutoModelForCausalLM = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(self.model_name, use_fast=False))
        object.__setattr__(self, 'model', AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        ).to(self.device))

    def _call(self, prompt: str, **kwargs) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=150, temperature=0.7, top_p=0.9)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @property
    def _identifying_params(self):
        return {"name_of_model": "phi-2"}

    @property
    def _llm_type(self) -> str:
        return "phi-2"

phi2_llm = Phi2LLM()

# Test Query

query = "What are the benefits of Phi-2 for chatbots?"
response = phi2_llm._call(query)

print(f"Response: {response}")