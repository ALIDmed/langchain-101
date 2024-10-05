import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key="bSeLMxtIkOyhnTaE_MZXPq6muuJn8s"
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_dir
)

def query_vector_store(store_name, query, embedding, search_type, search_kwargs):
    if not os.path.exists(persistent_dir):
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        relevant_docs = retriever.invoke(query)
        for i, doc in enumerate(relevant_docs):
            print(f"\n\nDocument{i}:\n{doc.page_content}\n\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"The vectore store {store_name} does not exist")

query = "How do you treat heteroscedasticity in regression models?"

############# Different retrieval methods ############# 

"""
1. Similarity Search
    - This method retrieves documents based on vector similarity.
    - It finds the most similar documents to the query vector based on cosine similarity.
"""
print("\n--- Using Similarity Search ---\n")
query_vector_store("chroma_db_with_metadata", query,
                   embeddings, "similarity", {"k": 3})