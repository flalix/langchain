import os

from dotenv import load_dotenv

# https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain_classic/document_loaders

from langchain_community.document_loaders import PyPDFLoader, TextLoader # , WhatsAppChatLoader, GoogleDriveLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore


root_data = '~/bioinformatics/LLM_Transformers_models_and_concepts'

load_dotenv()

if __name__ == "__main__":
    print("Ingesting")
    # print(os.environ["PINECONE_API_KEY"])

    fname = 'scGPT: toward building a foundation model for single-cell multi-omics using generative AI - 2024.pdf'
    filename = os.path.join(root_data, fname)

    try:

        if filename.endswith('.txt'):
            loader = TextLoader(filename, encoding='UTF-8')
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(filename)
        else:
            print("Define the doc type.")
            raise Exception('\n\n---------------- error stop ----------------\n\n')
    except:
        print(f"File {filename} does not exist.")
        raise Exception('\n\n---------------- error stop ----------------\n\n')

    
    # langchain document loaders
    document = loader.load()

    print('splitting document into chunks')

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n\n", length_function=len, is_separator_regex=False,)

    chuncks = text_splitter.split_documents(document)
    print(f"Created {len(chuncks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    llm = ChatOpenAI()

    print("Vector store link ...")
    print("Pinecone: vectors are stored in index name ", os.environ.get("INDEX_NAME"))


    PineconeVectorStore.from_documents(chuncks, embeddings, 
                                        index_name=os.environ.get("INDEX_NAME"))

    print(f"There are {len(chuncks)} chuncks, confirm at https://app.pinecone.io/organizations/-OYwN9xxzDa05svdS_Xl/projects/2d73f396-58f1-4a35-97bb-8972f5ae4b3a/indexes")
    print("Finish ...")
