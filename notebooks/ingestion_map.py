import asyncio
import os
import ssl
from typing import List, Any

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap  # TavilyCrawl

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, limit=1000)


def chunck_urls(urls: List[str], chunck_size:int=5) -> List[List[str]]:
    """
    Split URLs into chuncks of specified size.
    """
    chunks=[]
    for i in range(0, len(urls), chunck_size):
        chunk = urls[i:(i+chunck_size)]
        chunks.append(chunk)

    return chunks

async def extract_batch(urls: List[str], batch_num:int) -> List[dict[str, Any]]:
    '''
    Extract documents form a batch of URLs in asynchronous mode
    '''

    log_info(
        f"🗺️  TavilyExtract: processing batch {batch_num} with {len(urls)} URLs",
        Colors.BLUE,
    )

    try:
        docs = await tavily_extract.ainvoke(input={"urls": urls})

        n = len(docs.get("results", []))

        log_success(
            f"🗺️  TavilyExtract: completed batch {batch_num} and extracted {n} documents",
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION METHOD")
    log_info(
        f"🗺️  TavilyExtract: concurret extracton of {len(url_batches)} batches.",
        Colors.PURPLE,
    )

    tasks = [extract_batch(batch, i) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fuilter out exceptions and flatten results
    all_pages=[]
    failed_batches=0

    for result in results:
        if isinstance(result, Exception):
            log_error(f">>> TavilyExtract: batch failed with exeception - {result}")
            failed_batches+=1
        elif isinstance(result, list) and len(result) == 0:
            log_error(f">>> TavilyExtract: batch is empty - {result}")
            failed_batches+=1
        else:
            try:
                lista = result["results"]
            except Exception as e:
                print(">>>", result)
                log_error(f"TavilyExtract result is not a list of dict: {e}",)
                continue
                          
            for extracted_page in lista:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"🗺️  STavilyExtract: extration completed - total pages {len(all_pages)} and failed_batches {failed_batches}"
    )

    return all_pages

async def index_documents_async(documents: List[Document], batch_size: int=50):
    """
    Process documents in batch asynchronously
    """

    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents to the vector store.",
             Colors.DARKCYAN
             )
    
    #------- create batches ---------
    batches = [ documents[i:(i+batch_size)] for i in range(0, len(documents), batch_size)]

    log_info(
        f"🗺️  VectorStore Indexing: Split into {len(batches)} batches of {batch_size} batch size each document.",
        Colors.BLUE,
    )

    semaphore = asyncio.Semaphore(5)   # limit concurrency

    #------------ Process all batches concurrently ----------------
    async def add_batch(batch: List[Document], batch_num:int) -> bool:
        async with semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                log_success(
                    f"VectorStore Indexing: sucessfully added batch {batch_num}/{len(batches)} documents"
                )
                return True
            
            except Exception as e:
                log_error(f"VectorStore Indexing: failed to add batch {batch_num} - {e}")
                return False
            
    
    # Process batches concurrently
    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    #-------- count successful batches -------
    successful = sum([1 if result==True else 0 for result in results])

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: all batches Ok: {successful}/{len(batches)}"
        )
    else:
        log_warning(
            f"VectorStore Indexing: error processing batches: {successful}/{len(batches)}"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyMap: Starting to mp the documentation site https://python.langchain.com/",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    try:
        site_map = tavily_map.invoke("https://python.langchain.com/")

        log_success(
            f"🗺️  TavilyMap: Sucessfully mapped {len(site_map['results'])} URLS from documentaion site."
        )
    except Exception as e:
        print("!!! Error on mapping with TavilyMap", e)
        print(">>> site_map", site_map)
        return

    #Split URLs into batches of 5
    url_list = list(site_map["results"])

    print(">>>", len(url_list))
    for i, url in enumerate(url_list):
        print(i, url)

    url_batches = chunck_urls(url_list, chunck_size=5)
    log_info(
        f"🗺️  URL processing: split url_list {len(url_list)} into {len(url_batches)}",
        Colors.BLUE,
    )

    print(">>>", len(url_batches))
    for i, batch in enumerate(url_batches):
        print(i, len(batch))

    doc_list = await async_extract(url_batches)


    #------------ split the documents into str chunks ------------------
    chunk_size = 4000
    chunk_overlap = 200

    log_header("DOCUMENT CHUNCKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(doc_list) }documents with chunk_size={chunk_size} and chunk_overlap {chunk_overlap}",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    splited_docs = text_splitter.split_documents(doc_list)

    log_success(
        f"Text splitter: creeated {len(splited_docs)} chunks from {len(doc_list)} documents"
    )

    # Process the documents asynchronously
    await index_documents_async(splited_docs, batch_size=200)

    log_header("PINELINE IS COMPLETED")
    log_success(
        f"Documentation ingestion pipeline finished sucessfully!"
    )
    log_info(
        f"Summary:", Colors.BOLD
    )
    log_info(f"   - URLs mapped: {len(site_map['results'])}" )
    log_info(f"   - Documents extracted: {len(doc_list)}" )
    log_info(f"   - Chunks created: {len(splited_docs)}" )
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
