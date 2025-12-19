# app.py
import streamlit as st
import tempfile
from rag_engine import build_rag, ask_question
from dotenv import load_dotenv

# Force reload of .env file, overwriting any existing env vars
load_dotenv(override=True)


st.set_page_config(page_title="PDF Chatbot", layout="centered")
name = st.text_input("Please Type your name: ")
st.warning(f"Hey! {name}")
st.title("Chat with your PDF")

if "rag" not in st.session_state:
    st.session_state.rag = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.rag is None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Processing PDF..."):
        st.session_state.rag = build_rag(pdf_path)

    st.success("PDF loaded successfully!")
    st.balloons()

question = st.text_input("Ask a question")

if question and st.session_state.rag:
    retriever, llm = st.session_state.rag

    with st.spinner("Thinking..."):
        answer = ask_question(retriever, llm, question)

    st.markdown("Answer")
    st.write(answer)
