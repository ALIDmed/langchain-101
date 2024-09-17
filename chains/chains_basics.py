from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3", num_gpu=-1)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a senio {job}"),
        ("human", "tell {count} thing to do before applying for a {job} post")
    ]
)

chain = prompt_template | model | StrOutputParser()

res = chain.invoke({"job": "data scientist", "count": 5})
print(res)