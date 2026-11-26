from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
COLLECTION_NAME = "pdf_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

COMPANY_FILES = {
    "reliance": "RIL_AR 2026.pdf",
    "vedanta": "ved_AR.pdf",
}
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

