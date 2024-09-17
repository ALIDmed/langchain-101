from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

template = "tell something interesting about {topic}"
prompt_template = PromptTemplate.from_template(template)


prompt = prompt_template.invoke({"topic":"machine learning"})

print(prompt)