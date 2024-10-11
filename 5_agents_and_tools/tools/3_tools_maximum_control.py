import os
from typing import Type, Optional

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.callbacks import BaseCallbackManager

load_dotenv()

class SimpleSearchArgs(BaseModel):
    query: str = Field(description="should be a search query")

class MultiplyNumbersArgs(BaseModel):
    a: float = Field(description="first number to multiply")
    b: float = Field(description="second number to multiply")

class SimpleSearchTool(BaseTool):
    name: str = "simple_search"
    description: str = "Useful for when you need to answer questions about current events."
    args_schema: Type[BaseModel] = SimpleSearchArgs
    
    def _run(self, query: str) -> str:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key)

        results = client.search(query=query)
        return f"Search results for {query}\n\n{results}\n"
class MultiplyNumbersTool(BaseTool):
    name: str = "multiply_numbers"
    description: str = "useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(
        self,
        x: float,
        y: float,
    ) -> str:
        """Use the tool."""
        result = x * y
        return f"The product of {x} and {y} is {result}"


    def _run(self, a: float, b: float) -> str:
        return "the product of {a} and {b} is {a*b}"
    
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool()
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