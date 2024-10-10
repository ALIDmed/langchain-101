import os

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OllamaEmbeddings(model='llama3.1')

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':3, "score_threshold":0.25}
)

query = 'What is market basket analysis? How would you do it in Python?'

relevant_docs = retriever.invoke(query)

print("--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document{i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")