import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def greet_user(name: str) -> str:
    return f"Hello {name}"

def reverse_string(text: str) -> str:
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    return a + b

class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="first string")
    b: str = Field(description="second string")

tools = [
    Tool(
        name="GreetUser", 
        func=greet_user, 
        description="Greets the user by name.",
    ),
    Tool(
        name="ReverseString", 
        func=reverse_string, 
        description="Reverses the given string.",
    ),
    # Use StructuredTool for more complex functions that require multiple input parameters.
    # StructuredTool allows us to define an input schema using Pydantic, ensuring proper validation and description.
    StructuredTool.from_function(
        func=concatenate_strings, 
        name="ConcatenateStrings", 
        description="Concatenates two strings.",
        args_schema=ConcatenateStringsArgs,  # Schema defining the tool's input arguments
    ),
]
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

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)