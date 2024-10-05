import os
import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "PDFs", "Data Science Interview Preparation(DAY 1).pdf")
db_dir = os.path.join(current_dir, "db")

loader = PyMuPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


def create_vectore_store(docs, embeddings, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):

        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        db = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding=embeddings,
            persist_directory=persistent_dir
        )
    else:
        print(f"the vectore store {store_name} already exists.")

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key="api-key"
)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

create_vectore_store(docs, gemini_embeddings, "chroma_db_gemini")
create_vectore_store(docs, hf_embeddings, "chroma_db_HF")


db = Chroma(
    embedding_function=gemini_embeddings,
    persist_directory=os.path.join(db_dir, "chroma_db_gemini")
)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k':3, "score_threshold":0.3}
)

query = "What is the Variation Inflation Factor?"
relevant_docs = retriever.invoke(query)
for i, doc in enumerate(relevant_docs):
    print(f"Document{i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")