import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M: %p")

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="useful when you need to know the current time"
    )
]
# see the following prompt in : https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executer = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

response = agent_executer.invoke({"input": "what is the current time?"})
print("response", response)