from dotenv import load_dotenv
# import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    # print(">>> OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

if __name__ == "__main__":
    main()
