from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

model = OllamaLLM(model="llama3", num_gpu=-1)

chat_history = []
chat_history.append(
    SystemMessage(content="you are a helpful AI assisstant")
)

while True:
    message = input("Human: ")
    if message == "exit":
        break
    chat_history.append(
        HumanMessage(content=message)
    )

    result = model.invoke(chat_history)
    chat_history.append(
        SystemMessage(content=result)
    )

    print(f"AI: {result}")