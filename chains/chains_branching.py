from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

"""
Docs: https://python.langchain.com/docs/how_to/routing/#using-a-runnablebranch
"""
model = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token="api-key"
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
    ("system" "You are a helpful assistant. make sure to return the sentiment as well as the feedback"),
    ("human" ,"classify the sentiment of this feedback as positive, negative or neutral: {feedback}"),
])


#  list of (condition, runnable) pairs and a default runnable
branches = RunnableBranch(
    (
        lambda x: "positive" in x.lower(), 
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x.lower(), 
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x.lower(), 
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

review = "I was really disappointed with this movie"
res = chain.invoke(review)

print(res)
