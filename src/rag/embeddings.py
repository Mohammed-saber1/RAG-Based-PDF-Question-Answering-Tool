from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
from dotenv import load_dotenv
import os

class HuggingFaceEmbeddingsWrapper:
    """Wrapper for HuggingFace embeddings compatible with Chroma vector store."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_token: Optional[str] = None):
        """
        Initialize HuggingFace embeddings with a specified model.

        Args:
            model_name (str): Name of the HuggingFace model to use for embeddings. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
            hf_token (Optional[str]): Hugging Face API token for authenticated access. Defaults to None.
        
        Raises:
            RuntimeError: If initialization fails due to connection or authentication issues.
        """
        try:
            load_dotenv()  # Load environment variables from .env file
            # Set HF_TOKEN globally if provided or available in environment
            if hf_token or os.getenv("HF_TOKEN"):
                os.environ["HF_TOKEN"] = hf_token or os.getenv("HF_TOKEN")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Initialized HuggingFace embeddings with model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): List of document texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors for the documents.

        Raises:
            ValueError: If texts is empty or not a list of strings.
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input must be a non-empty list of strings")
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding vector for the query.

        Raises:
            ValueError: If text is empty or not a string.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        return self.embeddings.embed_query(text)