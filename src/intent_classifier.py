# Description: Module for intent classification using BAAI/bge-m3 model.

from typing import Optional, Dict, List
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# List of Intent Labels and Anchor Queries

INTENT_LABELS = {

    # ───────────── CONSUMER_DATA ───────────── #

    "user_profile": [
        "What plan type am I on — prepaid or postpaid?",
        "Is my account prepaid or postpaid?",
        "Tell me the type of subscription I have",
        "Am I subcribed to a postpaid plan?",
        "Am I subscribed to a prepaid plan?",
        "What is my registered city?",
        "Which city is my mobile number associated with?",
        "Tell me my account registration location",
        "What city is my account based in?",
        "Where was my account registered?",
        "Where is my SIM card registered?",
        "Show my profile: city and plan type",
        "Give me my user plan type and location"
    ],
    "cdrs_info": [
        "List my CDRs",
        "List my recent call records",
        "Give me an overview of my usage history",
        "What is my recent usage?",
        "Summarize my data, SMS, and call usage",
        "Show me my recent activity?",
        "How much data have I used?",
        "How many SMS have I sent?",
        "How many voice minutes have I used?",
        "How many resources have I used?",
        "What is my total resource consumption?",
        "Show me my recent consumption.",
        "Show my total consumption across all services",
        "How much have I spent on data?",
        "How much have I spent on SMS?",
        "How much have I spent on voice calls?",
        "How much have I spent in total on resources?",
        "What is my total spend on telecom services?",
        "List my total spend on data, SMS, and voice calls"
    ],
    "purchases_info": [
        "Show me my recent purchase history",
        "Show me my recent purchases",
        "What is my most recent purchase?",
        "List my recent transactions",
        "Summarize my recent transactions",
        "What have I bought recently?",
        "How many purchases have I made recently?",
        "What are my total purchases?",
        "How much have I spent on purchases?",
        "What is my total spend on purchases?"
    ],
    "tickets_info": [
        "List my support tickets",
        "Show me my complaint history",
        "Show me my recent support ticket records",
        "What complaints have I raised?",
        "Show me my latest support ticket",
        "What is the status of my last complaint?",
        "Give me an update on my recent support ticket status"
        "Was my latest support ticket resolved?"
    ],

    # ───────────── GENERAL_INSIGHTS ───────────── #
    
    "regional_popularity": [
        "Which city has the most telecom users?",
        "Where are the most users registed to?",
        "List the user count for each city.",
        "Regional breakdown of user base",
    ],
    "user_type_distribution": [
        "What is the overall account type distribution?",
        "How many users are postpaid vs prepaid?",
        "Which type of subscription is more popular?",
        "Are there more postpaid or prepaid users?",
        "List the total number of prepaid and postpaid subscribers",
        "How many users across the network are on postpaid plans?",
        "How many users across the network are on prepaid plans?",
        "What is the split between prepaid and postpaid users?",
        "What percentage of customers are postpaid?",
        "What percentage of customers are prepaid?",
    ],
    "regional_user_type_distribution": [
        "How many postpaid users are in each city?",
        "How many prepaid users are in each city?",
        "Which city has the most postpaid customers?",
        "Which city has the most prepaid customers?",
        "Which location has the most postpaid users?",
        "Which location has the most prepaid users?",
        "What type of subscription is popular in my area?",
        "List the number of postpaid subscribers in my area",
        "List the number of prepaid subscribers in my area",
        "How many postpaid subscribers does my city have?",
        "How many prepaid subscribers does my city have",
        "Doe my city have more postpaid or prepaid users?"
    ],
    "ticket_statistics": [
        "What are the most common ticket categories?",
        "List the most common types of support tickets",
        "How many total complaints have been logged for each category?",
        "What complaint categories are logged most frequently?",
        "What issue is raised most often?",
        "How frequently is this type of issue raised?",
        "How common is my issue?",
        "What are the different complaint categories?"
    ],
    "resolution_times": [
        "What is the average resolution time for each ticket category?",
        "List the average resolution time for complaints",
        "How much time is expected to resolve my support ticket?",
        "Give me the ETA for resolving my recent complaint",
        "How long do tickets usually take to resolve?",
        "How long will it take to fix my issue?",
        "What type of issue takes the longest to be resolved?"
    ],

    # ───────────── FALLBACK ───────────── #
    "other": [
        "Hi",
        "Help me",
        "I have a question",
        "I need some help",
        "Can you help me",
        "I require your assistance,",
        "Can you assist me?",
        "May I speak to a human representative?",
        "Connect me to human representative",
        "Talk to a human agent"
    ]
}

INTENT_TO_METADATA = {

    # ───────────── CONSUMER_DATA ───────────── #

   "user_profile": {
        "category": "consumer_data",
        "type": "User Data",
        "section": "user_profile"
    },
    "cdrs_info": {
        "category": "consumer_data",
        "type": "User Data",
        "section": "cdrs"
    },
    "purchases_info": {
        "category": "consumer_data",
        "type": "User Data",
        "section": "purchases"
    },
    "tickets_info": {
        "category": "consumer_data",
        "type": "User Data",
        "section": "tickets"
    },

    # ───────────── GENERAL_INSIGHTS ───────────── #
    
    "regional_popularity": {
        "category": "general_insights",
        "subcategory": "Regional Popularity"
    },
    "user_type_distribution": {
        "category": "general_insights",
        "subcategory": "User Type Distribution"
    },
    "regional_user_type_distribution": {
        "category": "general_insights",
        "subcategory": "Regional User Type Distribution"
    },
    "ticket_statistics": {
        "category": "general_insights",
        "subcategory": "Most Common Ticket Categories"
    },
    "resolution_times": {
        "category": "general_insights",
        "subcategory": "Average Resolution Time Per Ticket Category"
    }
}

# BAAI/bge-m3 Intent Classifier Class

class BGEIntentClassifier:
    """
    Intent Classifier that uses the BAAI/bge-m3 model via Sentence-Transformers.
    """

    def __init__(
        self,
        intent_labels: Dict[str, List[str]],
        intent_metadata_map: Dict[str, dict],
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu"
    ):
        self.intent_labels = intent_labels
        self.intent_metadata_map = intent_metadata_map
        self.model = SentenceTransformer(model_name, device=device)
        self.anchor_embeddings = self._encode_anchors()

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings using the loaded model.
        Returns a numpy array of shape (num_texts, embed_dim).
        """
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def _encode_anchors(self) -> Dict[str, np.ndarray]:
        """
        Embed all anchor examples and compute their average for each intent.
        """
        anchor_embeddings = {}
        for intent, examples in self.intent_labels.items():
            emb = self._embed(examples)
            mean_emb = emb.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-9:
                mean_emb /= norm
            anchor_embeddings[intent] = mean_emb
        return anchor_embeddings

    def classify(self, query: str) -> Optional[dict]:
        """
        Classify the intent of a single query string.

        Returns:
            A dict with:
              - "intent": str
              - "score": float (cosine similarity)
              - "metadata_filter": Additional metadata for that intent
        """
        query_emb = self._embed([query])[0]

        scores = {}
        for intent, anchor_emb in self.anchor_embeddings.items():
            sim_score = np.dot(query_emb, anchor_emb)
            scores[intent] = float(sim_score)

        top_intent = max(scores, key=scores.get)
        return {
            "intent": top_intent,
            "score": scores[top_intent],
            "metadata_filter": self.intent_metadata_map.get(top_intent)
        }