# Ask a question about the PDFs already stored in the vector store.
# Usage:  python ask.py "What was the profit before tax?"
# or just run  python ask.py  and type the question when asked.

import sys

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.pipeline import llm, rag_advanced


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ")

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore()
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)

    result = rag_advanced(question, rag_retriever, llm, top_k=10, return_context=True)

    print("\nAnswer:", result['answer'])
    print("Confidence:", result['confidence'])
    print("Sources:")
    for source in result['sources']:
        # print(f"  - {source['source']}, page {source['page']} (score {source['score']:.2f})")
        
        print(f"  - {source['source']}, page {source['page']} (score {source['score']:.2f}, rerank {source['rerank_score']:.2f})")


if __name__ == "__main__":
    main()
