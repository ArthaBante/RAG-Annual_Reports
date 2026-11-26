from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.config import RERANK_MODEL


class RAGRetriever:

    """Handles query based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, rerank_model: str = RERANK_MODEL):
        """
        Initialise the retriever

        Args:
            vector_store: Vector store containing doc embeddings
            embedding_manager: manager for generating query embeddings

        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.rerank_model_name = rerank_model
        self.reranker = CrossEncoder(self.rerank_model_name)


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0, source_file: str=None) -> List[Dict[str, Any]]:
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
            
            query_args = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
            }
            if source_file:
                query_args["where"] = {"source_file": source_file}

            results = self.vector_store.collection.query(**query_args)


            # results = self.vector_store.collection.query(
                
            #     query_embeddings = [query_embedding.tolist()],
            #     n_results=top_k
            # )

            
            ####process results

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


    def rerank_model(self, query: str, documents: List[Dict[str, Any]], keep: int = 5) ->List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, document['content']) for document in documents]

        scores = self.reranker.predict(pairs)

        for document, score in zip(documents, scores):
            document['rerank_score'] = float(score)

        reranked = sorted(documents, key=lambda x:x['rerank_score'], reverse=True)

        return reranked[:keep]



