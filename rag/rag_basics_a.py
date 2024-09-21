import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "PDFs", "Data Science Interview Preparation.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Initializing vectore store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"the file {file_path} does not exist.")
    
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    """
    CharacterTextSplitter: In a nutshell, it takes the content of a document and splits it by the default separator(\n\n) which is the first level of chunking. If the first level of split creates a chunk greater than the specified chunk size, it does not split it further. However, if the first level of splitting generates smaller size chunk(less than the specified chunk size), it attempts to merge it with another chunk to adhere to the specified chunk size. So, if the first level split generates chunks greater than the chunk size, this logic will be not be of much use. 
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(documents)

    print("#### Documents Chunks Info ####")
    print(f"the number of document chunks: {len(docs)}")
    print(f"Simple chunk: \n{docs[1].page_content}")

else:
    pass