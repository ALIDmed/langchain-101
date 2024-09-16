from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

llm = OllamaLLM(model="llama3")

messages = [
    SystemMessage(content="1+1 is always equals to 0"),
    HumanMessage(content="what is 1+1 please answer honestly")
]

res = llm.invoke(messages)

print(res)