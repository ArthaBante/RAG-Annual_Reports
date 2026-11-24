import os
import uuid
from typing import List, Any

import numpy as np
import chromadb

from src.config import COLLECTION_NAME, VECTORSTORE_DIR


class VectorStore:

    def __init__(self, collection_name: str = COLLECTION_NAME, persist_directory: str = str(VECTORSTORE_DIR)):

        """Initiate the vector store

        Args:
            collection_name = Name of the chromadb collection
            persist_directory =

        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()


    def _initialize_store(self):
        """
        Initialize chromadb client and collection

        """

        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path = self.persist_directory)

            # get or create collection

            self.collection = self.client.get_or_create_collection(

                name = self.collection_name,
                metadata = {"description":"PDF document for RAG", "hnsw:space":"cosine"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")


        except Exception as e:
            print(f"Error initializing store vector: {e}")
            raise



    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        add documents and their embeddings in the vector store

        Args:
            documents: List of langchain documents
            embeddings: corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("No. of documents must match the number of embeddings")

        print(f"Adding {len(documents)} documents to the vector store")

        # prepare data for chromadb

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            #generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index']= i
            metadata['content_length']= len(doc.page_content)
            metadatas.append(metadata)

            #documents content
            documents_text.append(doc.page_content)

            #embedding
            embeddings_list.append(embedding.tolist())

        #add to collection

        try:
            self.collection.add(
                ids = ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = documents_text
            )
            print(f"successfully added {len(documents)} document to the server")
            print(f"total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"error adding documents to the vector store: {e}")
            raise
