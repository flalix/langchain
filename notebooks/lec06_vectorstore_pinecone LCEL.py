import os

from dotenv import load_dotenv

# https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain_classic/document_loaders

from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader, TextLoader # , WhatsAppChatLoader, GoogleDriveLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from operator import itemgetter

load_dotenv()

print("Initializing components ...")

embeddings = OpenAIEmbeddings()

# llm = ChatOpenAI(model="gpt-5.2")
llm = ChatOpenAI()

print("Vector store link ...")
print("Pinecone: vectors are stored in index name ", os.environ.get("INDEX_NAME"))

vectorStore = PineconeVectorStore(embedding=embeddings, 
                                  index_name=os.environ.get("INDEX_NAME")
                                 )

retriever = vectorStore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer.
    """
)

def format_docs(docs: list) -> str:
    """
    Format retrieved documents into a single string.
    """
    return "\n\n".join( doc.page_content for doc in docs )


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, foramts them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async suport without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Steop 1: retrieve relavant documents
    # docs is a list of k doc
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: invoke the LLM with the formated messages
    response = llm.invoke(messages)

    return response


def create_retrievel_chain_with_lcel():
    """
    Create a retrieval chain using LCEL
    Returns a chain that can be invoked wity {"question": "..."}

    Advances over non-LCEL approach:
    - Declarative and composable: easy to chain operations iwth pipe operator
    - Built-in streaming: chain.stream() owrks ou of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: better integration with LangChain's type system
    - Less code: more concise and readable
    - Reusable: chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools

    Debug in LangSmith:
    https://smith.langchain.com/o/4881c1f7-1c4c-4016-86be-7e2757b06fe5/projects/p/a6d41c63-ccdd-4568-9432-c651b608d82b?timeModel=%7B%22duration%22%3A%221d%22%7D
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        ) |
        prompt_template |
        llm |
        StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving ...")

    #===========================================================================
    #==== Option 2: RAG implementation with LangChain Expression Language LCEL
    #===========================================================================

    print("\n" + "="*40)
    print("RAG implementation with LCEL")
    print("="*40)

    chain_with_lcel = create_retrievel_chain_with_lcel()

    query = "What is Pinecone in ML?"
    result = chain_with_lcel.invoke({"question": query})

    print("\nAnswer:")
    print(result)
    print("\n")

'''
gpt-3.5-turbo

Answer:
Pinecone is a cloud-based vector database that specializes in high-performance similarity search for machine learning applications. It provides fast and efficient querying capabilities for large-scale datasets, making it ideal for tasks such as recommendation systems, image and video search, and natural language processing. Pinecone uses advanced indexing techniques and data structures to store and retrieve vectors, allowing for real-time search and retrieval of similar items in high-dimensional spaces.


gpt-5.2

Answer:
Pinecone is a **managed vector database** commonly used in machine learning applications to **store, index, and search vector embeddings** efficiently.

### What it’s used for
In many ML systems you convert data (text, images, audio, etc.) into **embeddings**—high-dimensional numeric vectors produced by models like OpenAI, Sentence Transformers, or CLIP. Pinecone lets you:

- **Upload embeddings** (plus IDs and metadata)
- **Perform fast similarity search** (nearest-neighbor lookup) to find items “most similar” to a query embedding
- **Filter by metadata** (e.g., only search within a specific user, date range, category)

### Common ML / LLM use cases
- **RAG (Retrieval-Augmented Generation):** retrieve relevant documents to feed into an LLM for grounded answers
- **Semantic search:** search by meaning rather than keywords
- **Recommendation systems:** “users/items similar to this”
- **Deduplication / clustering:** detect near-duplicates or group similar content
- **Memory for agents/chatbots:** store and retrieve past interactions or facts

### Why people use it
- Optimized for **approximate nearest neighbor (ANN)** search at scale
- Handles **indexing, scaling, and performance** without you running your own infrastructure
- Supports **real-time updates** (insert/upsert/delete) and fast queries

If you tell me your use case (RAG, recommendations, semantic search, etc.), I can outline the typical architecture and what you’d store in Pinecone.

'''
