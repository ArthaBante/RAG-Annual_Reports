# Run this once (or when you add new PDFs) to fill the vector store.
# Running it again adds the same chunks a second time, so delete
# data/vectorstore first if you want a fresh start.

from src.config import DATA_DIR
from src.loader import process_all_pdf, split_documents
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


def main():
    all_pdf_documents = process_all_pdf(DATA_DIR)
    chunks = split_documents(all_pdf_documents)

    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore()

    ## converting the text into the embeddings
    text = [doc.page_content for doc in chunks]

    ## Generate the embeddings
    embeddings = embedding_manager.generate_embeddings(text)

    ##store in the vector database
    vectorstore.add_documents(chunks, embeddings)


if __name__ == "__main__":
    main()
