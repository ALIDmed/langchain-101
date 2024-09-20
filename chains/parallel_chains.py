from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    huggingfacehub_api_token="api-key"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}")
    ]
)

def pros_prompt(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "you are an expert product reviewer"),
        ("human", "given these features: {features}, list the pros of these features")
    ])
    return pros_template.format_prompt(features=features)

def cons_prompt(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "you are an expert product reviewer"),
        ("human", "given these features: {features}, list the cons of these features")
    ])
    return cons_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\nCons:\n{cons}"

pros_chain = RunnableLambda(lambda x: pros_prompt(x)) | model | StrOutputParser()
cons_chain = RunnableLambda(lambda x: cons_prompt(x)) | model | StrOutputParser()

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel({"pros": pros_chain, "cons": cons_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["pros"], x["cons"]))
)

res = chain.invoke({"product_name": "iphone X"})
print(res)