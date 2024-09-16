from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3")

result = model.invoke("why islam is the best and most relistic religion out there?")

print(result)
