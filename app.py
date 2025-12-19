# app.py
import streamlit as st
import tempfile
from rag_engine import build_rag, ask_question
from dotenv import load_dotenv

# Force reload of .env file, overwriting any existing env vars
load_dotenv(override=True)
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background-color: #161b22;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

/* User message */
[data-testid="stChatMessage"] .stMarkdown {
    color: #e6edf3;
}

/* Input box */
textarea {
    background-color: #0e1117 !important;
    color: #e6edf3 !important;
    border-radius: 8px;
}

/* Buttons */
button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
}
</style>
<script>
    window.addEventListener("beforeunload", function (e) {
        e.preventDefault();
        e.returnValue = "";
    });
</script>
""", unsafe_allow_html=True)
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(page_title="PDF Chatbot", layout="centered")

st.title("Chat with your PDF")

if "rag" not in st.session_state:
    st.session_state.rag = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
        st.success(f"Loaded: {uploaded_file.name}")

if uploaded_file and uploaded_file.name != st.session_state.current_pdf:
    # New PDF uploaded → reset everything
    st.session_state.current_pdf = uploaded_file.name
    st.session_state.messages = []
    st.session_state.rag = None
if uploaded_file and st.session_state.rag is None:
    with st.spinner("Processing PDF..."):
        retriever, llm = build_rag(uploaded_file)
        st.session_state.rag = (retriever, llm)


if uploaded_file and st.session_state.rag is None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Processing PDF..."):
        st.session_state.rag = build_rag(pdf_path)

    st.success("PDF loaded successfully!")
    st.balloons()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Kuch puchna hai?🙄.....")

if question and st.session_state.rag:
    retriever, llm = st.session_state.rag

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        answer = ask_question(
            retriever,
            llm,
            question,
            chat_history=st.session_state.messages
        )

    # Show assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)
