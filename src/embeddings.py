from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer ##embedding model is present in here

from src.config import EMBEDDING_MODEL


class EmbeddingManager:

    def __init__(self, model_name: str = EMBEDDING_MODEL):

        """
        here we are initialising the embedding manager

        Args:
            model_name = Hugging face model name for sentence embeddings

        """

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):

        """Load the sentence transformer model"""

        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully. Embedding dimension: {self.model.get_embedding_dimension()}")

        except Exception as e:
            print(f"Error loading model {self.model_name}:{e}")
            raise


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        """Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dimension)

        """
        if not self.model:
            raise ValueError("model not loaded")

        print(f"generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generate embeddings with shape: {embeddings.shape}")
        return embeddings
