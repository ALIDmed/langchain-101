import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=""
)

db = Chroma(
    embedding_function=embeddings,
    persist_directory=persistent_dir
)
query = "What is the difference between binomial distribution and polynomial distribution?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

relevant_docs = retriever.invoke(query)

for i, doc in enumerate(relevant_docs):
    print(f"Document{i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

prompt = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\n Relevant documents:\n\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n Please provide an answer based only on the provided documents, if the answer is not found reply by 'I don't know'"
)

llm  = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="")

messages = [
    SystemMessage(content="You are a helpful assisstant."),
    HumanMessage(content=prompt)
]

result = llm.invoke(messages)
print(result.content)