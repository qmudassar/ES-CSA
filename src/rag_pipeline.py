# Description: Module for complete RAG pipeline.

import re
import sqlite3
from typing import Optional, List
from langchain_core.documents import Document
from embedding_utils import LocalEmbedding
from intent_classifier import BGEIntentClassifier
from insight_extraction import compute_consumer_insights, compute_general_insights

# Extracting MSISDN

def parse_msisdn(query: str) -> Optional[str]:
    """
    Parses a valid MSISDN (10-15 digits) from query text.
    """
    match = re.search(r"\b(?:\+?\d{1,3})?\d{10,15}\b", query)
    if match:
        msisdn = match.group().lstrip("+")
        if msisdn.isdigit() and 10 <= len(msisdn) <= 15:
            return msisdn
    return None

def get_msisdn_from_context(query: str = None, session_context: dict = None) -> Optional[str]:

    if session_context and "msisdn" in session_context:
        return session_context["msisdn"].strip().lstrip("+")

    if query:
        return parse_msisdn(query)

    return None


# Resolving User Identity

def resolve_user_identity(msisdn: str, db_path: str = "msisdn_mapping.db") -> Optional[dict]:

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_index, user_name FROM msisdn_mapping WHERE msisdn = ?",
            (msisdn,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "user_index": result[0],
                "user_name": result[1],
                "msisdn": msisdn
            }
        return None
    except sqlite3.Error as e:
        print(f"[ERROR] SQLite query failed: {e}")
        return None
    
# Retrieving Relevant Documents

def retrieve_relevant_docs(
    query: str,
    classifier,
    user_info: dict,
    vectorstore,
    k: int = 4,
    fallback_threshold: float = 0.65
):
    """
    Classify query intent, resolve metadata filters, and retrieve user-specific context.
    Post-parses raw context into structured atomized facts for consumer_data and general_insights.
    """
    intent_result = classifier.classify(query)
    intent_label = intent_result["intent"]
    confidence = intent_result["score"]
    filters = intent_result.get("metadata_filter", {}) or {}

    if filters.get("category") == "consumer_data":
        filters["user_index"] = user_info["user_index"]

    fallback_triggered = intent_label == "other" or confidence < fallback_threshold
    wrapped_filter = {"$and": [{key: val} for key, val in filters.items()]} if filters else None

    if fallback_triggered:
        raw_results = []
    else:
        raw_results = (
            vectorstore.similarity_search(query, k=k, filter=wrapped_filter)
            if wrapped_filter else
            vectorstore.similarity_search(query, k=k)
        )

    if filters.get("category") == "consumer_data" and not fallback_triggered:
        processed_results = compute_consumer_insights(raw_results)
    elif filters.get("category") == "general_insights" and not fallback_triggered:
        processed_results = compute_general_insights(raw_results)
    else:
        processed_results = [doc.page_content for doc in raw_results]

    return {
        "query": query,
        "intent": intent_label,
        "filters": wrapped_filter,
        "user_info": user_info,
        "confidence": confidence,
        "results": processed_results,
        "fallback_triggered": fallback_triggered
    }

# Format Retrieved Context into Structured LLM Input

def format_context(intent: str, docs: List[str], user_info: dict) -> str:
    if not docs:
        return "No relevant context found for this request."

    user_name = user_info.get("user_name", "Unknown")
    msisdn = user_info.get("msisdn", "Unknown")
    user_identity = f"[User: {user_name} | MSISDN: {msisdn}]"

    raw_context = "\n".join(doc.strip() for doc in docs if doc.strip())

    title_map = {
        "user_profile": "User Profile",
        "cdrs_info": "CDRs",
        "purchases_info": "Purchase History",
        "tickets_info": "Support History",
        "regional_popularity": "Regional Popularity",
        "user_type_distribution": "User Type Distribution",
        "regional_user_type_distribution": "Regional User Type Distribution",
        "ticket_statistics": "Ticket Statistics",
        "resolution_times": "Average Ticket Resolution Times",
    }

    section_title = title_map.get(intent, "Account Information")

    formatted_input = (
        f"{user_identity}\n\n"
        f"{section_title}:\n"
        f"{raw_context.strip()}"
    )

    return formatted_input

# RAG Pipeline Wrapper

def run_rag_pipeline(
    query: str,
    session_context: dict,
    classifier,
    vectorstore,
    db_path: str = "msisdn_mapping.db",
    k: int = 1,
    fallback_threshold: float = 0.65,
):
    """
    Full RAG pipeline that resolves MSISDN from session,
    retrieves relevant context, and formats it for LLM input.
    """

    msisdn = get_msisdn_from_context(query=query, session_context=session_context)
    if not msisdn:
        return {
            "intent": "auth_required",
            "formatted": "Unable to resolve user identity. Please provide your number for authentication.",
            "user_info": ""
        }

    user_info = resolve_user_identity(msisdn, db_path=db_path)
    if not user_info:
        return {
            "intent": "user_not_found",
            "formatted": f"The SIM: {msisdn} was not found in the system.",
            "user_info": ""
        }

    results = retrieve_relevant_docs(
        query=query,
        classifier=classifier,
        user_info=user_info,
        vectorstore=vectorstore,
        k=k,
        fallback_threshold=fallback_threshold,
    )

    if results["fallback_triggered"] or not results["results"]:
        return {
            "intent": results.get("intent", "fallback"),
            "formatted": f"Directly Respond (No Context Available) – Query: {query}",
            "user_info": user_info
        }

    formatted = format_context(
        intent=results["intent"],
        docs=results["results"],
        user_info=results["user_info"]
    )

    return {
        "intent": results["intent"],
        "formatted": formatted,
        "user_info": results["user_info"]
    }

# Complete RAG Pipeline Wrapper with LLM Integration

def rag_pipeline_with_llm(
    query: str,
    session_context: dict,
    conversation_context: str,
    classifier,
    vectorstore,
    rag_chain,
    llm,
    db_path: str = "msisdn_mapping.db",
    k: int = 1,
    fallback_threshold: float = 0.65,
):
    result = run_rag_pipeline(
        query=query,
        session_context=session_context,
        classifier=classifier,
        vectorstore=vectorstore,
        db_path=db_path,
        k=k,
        fallback_threshold=fallback_threshold,
    )

    if isinstance(result, str) and "Directly Respond" in result:
      return llm.invoke(query)

    llm_response = rag_chain.invoke({
        "query": query,
        "intent": result["intent"],
        "docs": result["formatted"],
        "user_info": result["user_info"],
        "session_context": conversation_context
    })

    return llm_response