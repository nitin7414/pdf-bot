# rag_engine.py
import os
from dotenv import load_dotenv

# Force reload of .env file, overwriting any existing env vars
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def build_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="mistralai/devstral-2512:free",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    )

    return retriever, llm


def ask_question(retriever, llm, question):
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="Answer based only on the context below."),
        SystemMessage(content=context),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content
