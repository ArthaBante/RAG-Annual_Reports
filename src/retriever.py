from typing import List, Dict, Any

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


class RAGRetriever:

    """Handles query based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialise the retriever

        Args:
            vector_store: Vector store containing doc embeddings
            embedding_manager: manager for generating query embeddings

        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document for a query

        Args:
            query: The search query
            top_k: number of top results to return
            score_threshold: minimum similarity score threshold

            return:
                list of dictionaries containing retrieved documents and metadata
        """

        print(f"Retrieving documents for query: '{query}'")
        print(f" Top K: {top_k}, and Score Threshold: {score_threshold}")

        #Generate query embeddings

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        #search in vector store

        try:

            results = self.vector_store.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results=top_k
            )

            #process results

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    #convert distances to similarity score

                    similarity_score = 1-distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'ids': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score':similarity_score,
                            'distance':distance,
                            'rank':i+1
                        })

                print(f"retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise []
