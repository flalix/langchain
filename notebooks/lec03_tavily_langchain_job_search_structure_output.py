from dotenv import load_dotenv

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langchain_tavily import TavilySearch

'''
Structure output

https://docs.langchain.com/oss/python/langchain/structured-output

Structured output allows agents to return data in a specific, predictable format.
Instead of parsing natural language responses, you get structured data in the form of JSON objects, 
Pydantic models, or dataclasses that your application can use directly.

LangChain’s create_agent handles structured output automatically. 
The user sets their desired structured output schema, and when the model generates the structured data, 
it’s captured, validated, and returned in the 'structured_response' key of the agent’s state.

def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]

Provider Strategy
=================

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)


'''

class Source(BaseModel):
    '''
    Schema for a source usedby the agent
    '''
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    '''
    Schema for the agent response with an answer and sources
    '''

    answer: str = Field(description="The agent's answer to the question")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used by the agent to answer the question")

    # job_title: str = Field(description="The title of the job position")
    # company: str = Field(description="The company offering the job position")
    # location: str = Field(description="The location of the job position")
    # description: str = Field(description="A brief description of the job position")


load_dotenv()

# default: 'gpt-3.5-turbo-0125'
llm = ChatOpenAI()  # model='gpt-5'
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

question_task = "Search for any cutting-edge research job position on Computational Biology and Immunology and artificial intelligence in Europe and list their details."


def main():
    result = agent.invoke({"messages": [HumanMessage(content=question_task)]})
    print(result)
 

if __name__ == "__main__":
    main()
