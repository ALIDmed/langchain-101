from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_ollama.llms import OllamaLLM

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a senio {job}"),
        ("human", "tell {count} thing to do before applying for a {job} post")
    ]
)

print(prompt_template.format_prompt(**{"job": "x", 'count':2}))