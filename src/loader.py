from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


### Data ingestion - To read all the pdf's in a directory

def process_all_pdf(pdf_directory):

    #process all the pdf's in the directory

    all_documents = []
    pdf_dir = Path(pdf_directory)

    #find all the pdf files

    pdf_files = list(pdf_dir.glob("**/*.pdf")) #here we are making a list of all the pdf files

    print(f"found {len(pdf_files)} pdf files to process")

    ###now we're making the document out of the pdf files and also adding few extra meta data in it

    for pdf_file in pdf_files:

        print(f"\nProcessing: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            #adding some more meta data in the documents

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"loaded {len(documents)} pages")

        except Exception as e:
            print(f"error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


## now we will chunk these documents into smaller parts to perform better in a rag application

def split_documents(documents, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]

    )

    split_document = text_splitter.split_documents(documents)
    print(f"SPlit {len(documents)} documents into {len(split_document)} chunks")

    if split_document:
        print(f"\nExample Chunk")
        print(f"content: {split_document[0].page_content[:200]}...")
        print(f"Metadata: {split_document[1].metadata}")

    return split_document
