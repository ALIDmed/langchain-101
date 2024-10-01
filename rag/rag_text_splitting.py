import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")


if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"the file {file_path} doesn't exist"
    )

loader = TextLoader(file_path)
documents = loader.load()

embeddings = OllamaEmbeddings(model="llama3.1")

def create_vectore_store(docs, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print(f"Creating vector store {store_name}...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persistent_dir
        )
        print(f"Finished Creating vector store {store_name}...")
    else:
        print(f"vectore store {store_name} already exists")

