import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db_firecrawl")

def create_vector_store(query):
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("Firecrawl api key does not exist.")
    
    print("Crawling the website...")
    loader = FireCrawlLoader(
        url="https://eldenring.wiki.fextralife.com/Curseblade+Meera",
        api_key=api_key,
        mode="scrape"
        )
    docs = loader.load()
    print("Crawling finished")

    # convert metadata values to string if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_dir)
    retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
    )

    relevant_docs = retriever.invoke(query)
    for i, doc in enumerate(relevant_docs):
        print(f"\n\nDocument{i}:\n{doc.page_content}\n\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

query = "What is the FP cost of Curseblade Meera?"
create_vector_store(query)