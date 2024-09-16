from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage(content="You are a senior experienced python developper"),
    HumanMessage(content="write a python script to automate email sending")
]
# claude_llm = ChatAnthropic(model='claude-3-opus-20240229')

# res = claude_llm.invoke(messages)
# print(res)

gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
res = gemini_llm.invoke(messages)
print(res)
