# Load documents from knowledge_base/ and split into chunks

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import Config
import os

def load_and_chunk():
    docs = []

    # Load every .docx file from knowledge_base/
    for filename in os.listdir("knowledge_base/"):
        if filename.endswith(".docx"):
            path = os.path.join("knowledge_base", filename)
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents")

    # Split and chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")
    return chunks


