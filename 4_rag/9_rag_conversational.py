import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key="api-key"
)

db = Chroma(embedding_function=embeddings, persist_directory=persistent_dir)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="api-key")

contextualize_q_system_prompt = """
    Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
"""
what create_history_aware_retriever does is instead of feeding the chat_history and the user query to the retrieval to get the relevant documents which The retriever might not recognize the full meaning or intent behind these vague references. it reformulates this context-dependant question into a clear query it uses an llm to ensure that the question becomes complete and coherent without needing the retriever to "understand" the entire chat history it would turn for example "why would i use t-sne?" into "What are the benefits of using t-SNE for dimensionality reduction? and it does return the relevant docs.
Also Passing the entire chat history every time might lead to redundancy or irrelevant portions of the history being included
If you were to perform retrieval.invoke directly with the entire chat history appended to the query, the retriever would have to process a larger, more complex input every time"
"""
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n
    {context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
"""
create_stuff_documents_chain creates a chain for passing a list of documents to a model
"""
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
"""
This chain takes in a user inquiry, which is then passed to the retriever (in this case is history_aware_retriever) to fetch relevant documents. Those documents (and original inputs) are then passed to an LLM to generate a response
"""
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def rag_chat():
    print("Start chating with the AI. type 'exit' to end the conversation")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(query))
        chat_history.append(AIMessage(result['answer']))

if __name__ == "__main__":
    rag_chat()