from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model="llama3", num_gpu=-1)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a senio {job}"),
        ("human", "tell {count} thing to do before applying for a {job} post")
    ]
)

upper_output = RunnableLambda(lambda x: x.upper())
count_characters = RunnableLambda(lambda x: f'the len is {len(x)}\n text:\n{x}')

chain = prompt_template | model | StrOutputParser() | upper_output | count_characters

res = chain.invoke({"job": "data scientist", "count": 2})
print(res)
