import os, ssl
from typing import Any, Dict, List

import asyncio
import certifi

from dotenv import load_dotenv

# https://app.tavily.com/home
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma vector store (localy)
# Pinecone -> vector store cloud
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure the SSL context to use certifi certifications
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST CA_BUNDLE"] = certifi.where()

print("OpenAIEmbeddings ...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True,
                              chunk_size=50, retry_min_seconds=10)

print("Vector store link ...")
vectorStore = PineconeVectorStore(embedding=embeddings, 
                                  index_name=os.environ["INDEX_NAME"])
 
print("Tavily ...")
tavily_extract = TavilyExtract()
tavily_crawl   = TavilyCrawl()
tavily_map     = TavilyMap(max_depth=5, max_breath=20, max_pages=1000)


async def main():
    '''
    Main async function to orchestrate the entire process
    '''

    log_header('DOCUMENTATION INGESTION PIPELINE')

    log_info("TavilyCrawl: starting to Crawl", Colors.PURPLE)

    # https://docs.tavily.com/documentation/api-reference/endpoint/crawl
    print("tavily crawl")

    '''
    Crawl the documentation site - Tavily max depth
    start: max_depth = 1, 2
    after test ... increase
    '''

    URL = "https://python.langchain.com/"

    res = tavily_crawl.invoke({
        "url": URL,
        "max_depth": 1,
        "extract_depth": "advanced",
        # "instructions": "content on ai agents"
    })

    result_list = res["results"]
    
    doc_list = [Document(page_content=result['raw_content'], metadata={"source":result['url']}) for result in result_list]
    
    log_success(f"TavilyCrawl: successfully crawled {len(result_list)} -> doc_list {len(doc_list)} URL from {URL}")

    print("---- end ----")




if __name__ == "__main__":
    print("Initializing asyncio ...")

    asyncio.run(main())

 