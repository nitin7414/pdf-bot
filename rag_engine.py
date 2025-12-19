
import os
from dotenv import load_dotenv
import tempfile
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv(override=True)
def build_rag(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
    model="nomic-ai/nomic-embed-text-v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
    model="mistralai/devstral-2512:free",
    temperature=0.2,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
    )


    return retriever, llm



def ask_question(retriever, llm, question, chat_history):
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(
            content="You are a helpful assistant. "
                    "Answer ONLY using the context below. "
                    "If the answer is not in the context, say you don't know."
        ),
        SystemMessage(content=context)
    ]

    # Add last N messages for memory
    for msg in chat_history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content

