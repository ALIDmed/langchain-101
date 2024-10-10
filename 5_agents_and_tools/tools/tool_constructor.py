import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

def greet_user(name: str) -> str:
    return f"Hello {name}"

def reverse_string(text: str) -> str:
    return text[::-1]

def concatenate_text(a: str, b:str) -> str:
    return a + b

class ConcatenateStringArgs(BaseModel):
    a: str = Field(description="first string")
    b: str = Field(description="second string")


