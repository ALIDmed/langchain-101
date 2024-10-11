import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class SimpleSearchArgs(BaseModel):
    query: str = Field(description="should be a search query")

class SimpleSearchTool(BaseTool):
    name: str = "simple search"
    description: str = "Useful for when you need to answer questions about current events."
    args_schema: Type[BaseModel] = SimpleSearchArgs

    def _run(self, query: str) -> str:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key)

        results = client.search(query=query)
        return f"Search results for {query}\n\n{results}\n"
    
class MultiplyNumbersArgs(BaseModel):
    a: float = Field(description="first number to multiply")
    b: float = Field(description="second number to multiply")

class MultipleNumbersTool(BaseTool):
    name: str = "multiply numbers"
    description: str = "Useful for when you need to multiply two numbers"
    args_schema: type[BaseModel] = MultiplyNumbersArgs

    def _run(self, a: float, b:float) -> str:
        res = a * b
        return "the product of {a} and {b} is {res}"
    
tools = [
    SimpleSearchTool,
    MultipleNumbersTool
]

prompt = hub.pull("hwchase17/openai-tools-agent")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
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
response = agent_executor.invoke({"input": "Search for the new flux model"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 0.25 and 3"})
print("Response for 'Multiply 0.25 and 3':", response)