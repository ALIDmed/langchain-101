import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db_elding_ring")

urls = ["https://eldenring.wiki.fextralife.com/Curseblade+Meera"]
loader = WebBaseLoader(urls)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=""
)

if not os.path.exists(persistent_dir):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
else:
    db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

query = "Where can i find Curseblade Meera"
relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs):
    print(f"Document{i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")