# Description: Module for implementing Chat Session object to query the agent.

import torch
import textwrap
import sqlite3
from torch import bfloat16
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from embedding_utils import LocalEmbedding
from intent_classifier import BGEIntentClassifier, INTENT_LABELS, INTENT_TO_METADATA
from rag_pipeline import run_rag_pipeline, rag_pipeline_with_llm
from llm_pipeline import create_pipeline
from llm_wrapper import CoherePromptWrapper
from agent_init import initialize_agent

# Implementing Chat Session Object

class ChatSession:
    def __init__(self, msisdn: str):
        self.msisdn = msisdn
        self.history = []
        self.memory = ConversationBufferMemory(return_messages=True)

    def append(self, user_msg, bot_response):
        self.history.append((user_msg, bot_response))
        self.memory.chat_memory.add_user_message(user_msg)
        self.memory.chat_memory.add_ai_message(bot_response)

    def get_session_context(self):
        return {"msisdn": self.msisdn}
    
    def get_langchain_context(self, current_query: str) -> str:
        history_messages = self.memory.load_memory_variables({})["history"]
        return f"{history_messages}\nUser: {current_query}"

def chat_with_bot(
    session: ChatSession,
    query: str,
    classifier,
    vectorstore,
    rag_chain,
    llm,
    db_path="msisdn_mapping.db",
    k=1,
    fallback_threshold=0.65
):
    session_context = session.get_session_context()
    conversation_context = session.get_langchain_context(query)

    response = rag_pipeline_with_llm(
        query=query,
        session_context=session_context,
        conversation_context=conversation_context,
        classifier=classifier,
        vectorstore=vectorstore,
        rag_chain=rag_chain,
        llm=llm,
        db_path=db_path,
        k=k,
        fallback_threshold=fallback_threshold
    )

    session.append(query, response)

    return response

def run_chat_session(msisdn, vectorstore, classifier, rag_chain, llm):

    session = ChatSession(msisdn=msisdn)

    print("=" * 43)
    print(f"Welcome to Evamp & Saanga Customer Support!")
    print("=" * 43)
    print(
        "\nOur AI assistant is here to help you with personalized information and support.\n"

        "\n(To end your session at any time, type 'exit' or 'bye')\n"
    )
    print("How can we assist you today?")
    print("-" * 28)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit", "end", "bye", "goodbye"}:
            print("\nAssistant: Thank you for reaching out to Evamp & Saanga Support. If you have any more questions, feel free to return anytime. Have a wonderful day!")
            break

        response = chat_with_bot(
            session=session,
            query=query,
            classifier=classifier,
            vectorstore=vectorstore,
            rag_chain=rag_chain,
            llm=llm
        )

        label = "Assistant:"
        prefix = " " * len(label)
        wrapped_response = textwrap.fill(
            response.strip(),
            width=120,
            initial_indent=f"{label} ",
            subsequent_indent=f"{prefix} "
        )
        print("")
        print("-" * 120)
        print(f"{wrapped_response}")
        print("-" * 120)
