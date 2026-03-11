from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch
load_dotenv()

tools = [TavilySearch()]

'''
Agentic agents are autonomous entities that can perceive their environment,
make decisions, and take actions to achieve specific goals.
They are designed to operate independently, often using artificial intelligence techniques
to process information and interact with their surroundings.

@tool --> decorator to define a tool that an agent can use. 
Tools are functions or methods that perform specific tasks, such as searching the web, accessing a database, or performing calculations. Agents can call these tools to accomplish their goals.

Agent
-----------------------
    [ LLM ]

    Toolkit
   [ set of tools ]
-----------------------


Tool: is a function/method that performs a specific task, such as searching the web, 
accessing a database, or performing calculations. 

Tools are used by agents to accomplish their goals.  

'''

llm = ChatOpenAI(model='gpt-5')
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": [HumanMessage(content="Who is studying lncRNA and cancer?")]})
    print(result)
 

if __name__ == "__main__":
    main()
