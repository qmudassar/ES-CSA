# Description: This file contains the custom embedding wrapped to load in the SentenceTransformer model.

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class LocalEmbedding(Embeddings):
    """
    LangChain-compatible wrapper for SentenceTransformer models.
    Bypasses HuggingFaceEmbeddings and avoids 'transformers' dependency conflicts.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the SentenceTransformer model.

        Args:
            model_name (str): Name of the pre-trained sentence-transformers model.
            device (str): 'cpu' or 'cuda'
        """
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents (used for indexing).

        Args:
            texts (list[str]): List of documents.

        Returns:
            list[list[float]]: Embeddings
        """
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query (used for retrieval).

        Args:
            text (str): Input query.

        Returns:
            list[float]: Embedding
        """
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
