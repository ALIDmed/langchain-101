import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
pdfs_dir = os.path.join(current_dir, "PDFs")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

if not os.path.exists(persistent_directory):
    print("Initializing vectore store...")

    if not os.path.exists(pdfs_dir):
        raise FileNotFoundError(f"the directory {pdfs_dir} does not exist.")
    
    documents = []
    for file in os.listdir(pdfs_dir):
        file_path = os.path.join(pdfs_dir, file)
        loader = PyMuPDFLoader(file_path)
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata = {"source": file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("#### Documents Chunks Info ####")
    print(f"the number of document chunks: {len(docs)}")
    print(f"Simple chunk: \n{docs[1].page_content}")

    print('#### generating embeddings ####')
    embeddings = OllamaEmbeddings(model='llama3.1')

    print('#### creating vectore store ####')
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    print('#### Vectore store creation finished 🔥')

else:
    print("the vectore store already exists")