from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain_core.output_parsers import StrOutputParser

"""
Docs: https://python.langchain.com/docs/how_to/routing/#using-a-runnablebranch
"""
model = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token="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"
)

positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system" "You are a helpful assistant."),
    ("human" ,"Generate a thank you note for this positive feedback:{feedback}.") ,
])
negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system" "You are a helpful assistant."),
    ("human" ,"Generate a response addressing this negative feedback:{feedback}.") ,
])
neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system" "You are a helpful assistant."),
    ("human" ,"Generate a request for more details for this neutral feedback:{feedback}.") ,
])
escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human","Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
)

classification_template = ChatPromptTemplate.from_messages([
    ("system" "You are a helpful assistant."),
    ("human" ,"classify the sentiment of this feedback as positive, negative and neutral: {feedback}"),
])


#  list of (condition, runnable) pairs and a default runnable
branches = RunnableBranch(
    (
        lambda x: "positive" in x, 
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x, 
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x, 
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

review = "I absolutely loved this movie! From start to finish, it kept me hooked with its incredible storytelling and captivating performances."
res = chain.invoke(review)

print(res)
