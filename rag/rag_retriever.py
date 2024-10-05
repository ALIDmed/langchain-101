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
query_vector_store("chroma_db_with_metadata", 
                   query,
                   embeddings, 
                   search_type="similarity", 
                   search_kwargs={"k": 3}
                   )

"""
2. Similarity Score Threshold
    - This method retrieves documents that exceed a certain similarity score threshold.
    - 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
    - Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
"""
query_vector_store("chroma_db_with_metadata", 
                   query,
                   embeddings, 
                   search_type="similarity_score_threshold", 
                   search_kwargs={"k": 3, "score_threshold": 0.2}
                   )


"""
3. Max Marginal Relevance (MMR)
    - This method balances between selecting documents that are relevant to the query and diverse among themselves.
    - 'fetch_k' specifies the number of documents to initially fetch based on similarity.
    - 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
    - Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
    - Note: Relevance measures how closely documents match the query.
    - Note: Diversity ensures that the retrieved documents are not too similar to each other, providing a broader range of information.
"""
query_vector_store("chroma_db_with_metadata", 
                   query,
                   embeddings, 
                   search_type="mmr", 
                   search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
                   )