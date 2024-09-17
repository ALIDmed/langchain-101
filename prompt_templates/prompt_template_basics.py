from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

print("------Prompt from template------")
template = "tell something interesting about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic":"machine learning"})

print(prompt)

print("------Prompt template with system and human messages------")
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a senior {job}"),
    ("user", "tell me a {fact_count} facts about {job}"),
])

prompt = prompt_template.invoke({"job": "software engineer", "fact_count": 2})
print(prompt)