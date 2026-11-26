###simple RAG pipeline with Groq LLM
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL
from src.config import COMPANY_FILES




load_dotenv()

#Initialize the Groq LLM
google_api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=LLM_MODEL, temperature = 0.1, max_output_tokens =1024)


def detect_company_file(query):
    q = query.lower()
    for company, file_name in COMPANY_FILES.items():
        if company in q:
            return file_name
    return None


##2. Simple RAG function: retrieve context + generate response

# def rag_simple(query, retriever, llm, top_k = 3):
#     ##retrieve the context
#     results = retriever.retrieve(query, top_k=top_k)
#     context = "\n\n".join([doc['content'] for doc in results]) if results else ""
#     if not context:
#         return "NO relevant context found to answer the question"

#     ##Generate the answer using google llm

#     prompt = f"""Use the following context to answer the question precisely

#         Context:
#         {context}

#         Question:
#         {query}

#         Answer: """

#     response = llm.invoke([prompt.format(context=context, query = query)])
#     return response.text


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context = False, fetch_k = 25):
    """RAG pipeline with some extra features.
    Returns: Answers, sources, confidence score, and full context
        """
    source_file = detect_company_file(query)
    results = retriever.retrieve(query, top_k=fetch_k, score_threshold=min_score, source_file=source_file)
    #results = retriever.retrieve(query, top_k=top_k, score_threshold = min_score)

    results = retriever.rerank_model(query, results, keep = top_k)


    if not results:
        return{'answer':'No relevant information found', 'sources':[], 'confidence':0.0, 'context':''}

    #Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])

    sources = [{
        'source':doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page':doc['metadata'].get('page', 'unknown'),
        'score':doc['similarity_score'],
        'preview':doc['content'][:300]+'....'
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])


    #Generate answers
    prompt=f"""You are answering questions about company annual reports.
Use ONLY the context below. Do not use outside knowledge.
If the answer is not in the context, reply: "Not found in the report."
If the context only covers part of the company (for example one business unit), say so.
Quote figures exactly as written in the context, with their units. \nContext: \n{context}\n\nQuestion:\n{query}\n\nAnswers:"""
    repsonse=llm.invoke(prompt)

    output = {
        'answer':repsonse.text,
        'sources':sources,
        'confidence':confidence
    }
    if return_context:
        output['context'] = context
    return output
