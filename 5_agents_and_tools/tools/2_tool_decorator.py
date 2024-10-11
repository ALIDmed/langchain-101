import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello {name}"

class ReverseStringArgs(BaseModel):
    text: str = Field(description="text to be reversed")

@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="first string")
    b: str = Field(description="second string")

@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

tools = [greet_user, reverse_string, concatenate_strings]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)