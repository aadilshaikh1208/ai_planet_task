# Search the vector store and return relevant chunks

from rag.vector_store import get_vector_store
from utils.config import Config

def retrieve(query):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=Config.TOP_K)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:")

    for i, doc in enumerate(results):
        print(f"[{i+1}] Source: {doc.metadata.get('source', 'unknown')}")
        print(f"     Preview: {doc.page_content[:150]}\n")

    return results