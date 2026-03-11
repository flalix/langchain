from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

'''
Agentic agents are autonomous entities that can perceive their environment, make decisions, 
and take actions to achieve specific goals. 
They are designed to operate independently, often using artificial intelligence techniques 
to process information and interact with their surroundings.

from langchain.tools import tool

@tool --> decorator to define a tool that an agent can use. 
Tools are functions or methods that perform specific tasks, 
such as searching the web, accessing a database, or performing calculations. 
Agents can call these tools to accomplish their goals.

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

@tool
def search(query: str) -> str:
    '''
    Tool that searcher over the interent
    Args:
        query (str): the search query
    Returns:
        str: the search results
    '''
    print(f"Searching for: {query}")
    return tavily.search(query=query)


# default: 'gpt-3.5-turbo-0125'
llm = ChatOpenAI()  # model='gpt-5'
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    message = "Search for any cutting-edge research job position on Computational Biology and artificial intelligence in Europe and list their details."
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    print(result)
 

if __name__ == "__main__":
    main()
