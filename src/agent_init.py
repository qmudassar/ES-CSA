# Description: Module for Agent Initialization for Chat Session.

import torch
import sqlite3
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from embedding_utils import LocalEmbedding
from intent_classifier import BGEIntentClassifier, INTENT_LABELS, INTENT_TO_METADATA
from rag_pipeline import run_rag_pipeline, rag_pipeline_with_llm
from llm_pipeline import create_pipeline
from llm_wrapper import CoherePromptWrapper

# Agent Initialization Function

def initialize_agent():

    # Initializing MSISDN Mapping Database Path

    db_path = "msisdn_mapping.db"
    conn = sqlite3.connect(db_path)

   # Initializing Embedding Model (all-MiniLM-L6-v2)

    embedding_model = LocalEmbedding()

    # Initializing Vector Store (ChromaDB)

    vectorstore = Chroma(
        persist_directory="embeddings/chromadb_embeddings",
        embedding_function=embedding_model,
        collection_name="consumer_db"
    )

    # Initializing Intent Classifier (BAAI/bge-m3)
    
    classifier = BGEIntentClassifier(
    intent_labels=INTENT_LABELS,
    intent_metadata_map=INTENT_TO_METADATA
    )

    # Initialize Tokenizer & LLM Model (command-r7b-12-2024)

    model_id = "CohereForAI/c4ai-command-r7b-12-2024"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Initialize Text Generation Pipeline

    llm_pipeline = create_pipeline(model=model, tokenizer=tokenizer)

    # Wrap Pipeline using CoherePromptWrapper
    
    llm = CoherePromptWrapper(pipeline=llm_pipeline, tokenizer=tokenizer)

    return conn, embedding_model, vectorstore, classifier, llm
