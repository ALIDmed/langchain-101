from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3", num_gpu=-1)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a senio {job}"),
        ("human", "tell {count} thing to do before applying for a {job} post")
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x)

chain = RunnableSequence(
    first=format_prompt,
    middle=[invoke_model],
    last=parse_output
)

res = chain.invoke({"job": "software engineer", "count":3})
print(res)