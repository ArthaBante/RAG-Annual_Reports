###simple RAG pipeline with Groq LLM
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.config import LLM_MODEL

load_dotenv()

#Initialize the Groq LLM
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model_name=LLM_MODEL, temperature = 0.1, max_tokens =1024)


##2. Simple RAG function: retrieve context + generate response

def rag_simple(query, retriever, llm, top_k = 3):
    ##retrieve the context
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "NO relevant context found to answer the question"

    ##Generate the answer using groq llm

    prompt = f"""Use the following context to answer the question precisely

        Context:
        {context}

        Question:
        {query}

        Answer: """

    response = llm.invoke([prompt.format(context=context, query = query)])
    return response.content


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context = False):
    """RAG pipeline with some extra features.
    Returns: Answers, sources, confidence score, and full context
        """
    results = retriever.retrieve(query, top_k=top_k, score_threshold = min_score)
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
    prompt=f"""Use the following context to answer the question properly and concisely. \nContext: \n{context}\n\nQuestion:\n{query}\n\nAnswers:"""
    repsonse=llm.invoke([prompt.format(context=context, query=query)])

    output = {
        'answer':repsonse.content,
        'sources':sources,
        'confidence':confidence
    }
    if return_context:
        output['context'] = context
    return output
