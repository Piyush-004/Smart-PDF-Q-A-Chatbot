import os
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from google.generativeai import GenerativeModel
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from typing import Optional, List
from htmlTemplates import css
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

INITIAL_PROMPT = (
    "You are an expert assistant in helping user answer questions based on the document provided by them."
    "Answer the questions based on the provided document. " 
    "You can also have a friendly conversation with the user - acknowledge their questions and greet them."
    "Be concise and accurate."
)

class GeminiLLM(LLM):
    model: str = "models/gemini-1.5-flash"
    initial_prompt: str = INITIAL_PROMPT

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        full_prompt = f"{self.initial_prompt}\n\n{prompt}"
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        return response.text
    
    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/embedding-001", api_key: str=None):
        self.model_name = model_name
        self.api_key = GEMINI_API_KEY
        genai.configure(api_key = self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model = self.model_name,
                content = text,
                task_type = "retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        result = genai.embed_content(
            model = self.model_name,
            content = text,
            task_type = "retrieval_query"
        )
        return result["embedding"]

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("PDF not processed yet.")
        return

    with st.chat_message("user"):
        st.markdown(user_question)

    response = st.session_state.conversation.invoke({"question": user_question})
    bot_msg = response["answer"]
    st.session_state.chat_history = response["chat_history"]

    with st.chat_message("assistant"):
        st.markdown(bot_msg)

def get_raw_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(chunks):
    embedding = GeminiEmbeddings()
    return FAISS.from_texts(chunks, embedding = embedding)

def get_conversation_chain(embeddings, initial_prompt = INITIAL_PROMPT):
    llm = GeminiLLM(initial_prompt = initial_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, embeddings.as_retriever(), memory=memory)

def main():
    st.write(css, unsafe_allow_html=True)

    # ---- State ----
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # may become LangChain messages later
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False

    # ---- Header ----
    st.header("Smart PDF Assistant 🤖")

    # ---- Render chat history first (so input stays at the bottom) ----
    history = st.session_state.chat_history or []
    for msg in history:
        # Support both dicts like {"role": "...", "content": "..."} and LangChain messages
        if isinstance(msg, dict):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
        else:
            # LangChain messages typically expose .type in {"human","ai","system"} and .content
            msg_type = getattr(msg, "type", None) or msg.__class__.__name__.lower()
            if msg_type == "human":
                role = "user"
            elif msg_type == "ai":
                role = "assistant"
            else:
                role = "assistant"
            content = getattr(msg, "content", "")
        if content:
            with st.chat_message(role):
                st.markdown(content)

    # ---- Chat input at the bottom (auto-clears on submit) ----
    user_question = st.chat_input("Ask about your PDFs…")
    if user_question:
        # Delegate rendering and history updates to your existing handler.
        # It already shows the user + assistant messages and updates st.session_state.chat_history.
        handle_userinput(user_question)

    # ---- Sidebar ----
    with st.sidebar:
        st.subheader("Upload PDF(s)")
        pdf_docs = st.file_uploader(
            "Drop one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can add more later and re-process."
        )

        # File summary
        if pdf_docs:
            with st.expander(f"Selected files ({len(pdf_docs)})", expanded=False):
                for f in pdf_docs:
                    size_kb = getattr(f, "size", 0) / 1024 if getattr(f, "size", None) else None
                    size_txt = f" • {size_kb:.1f} KB" if size_kb else ""
                    st.write(f"- {f.name}{size_txt}")

        # Process button
        process_disabled = not pdf_docs
        if st.button("Fetch and Process PDFs", disabled=process_disabled, use_container_width=True):
            if not pdf_docs:
                st.info("Add at least one PDF to begin.")
            else:
                with st.spinner("Processing PDFs… This can take a moment for large files."):
                    raw_text = get_raw_text(pdf_docs)
                    chunks = get_chunks(raw_text)
                    embeddings = get_embeddings(chunks)
                    st.session_state.conversation = get_conversation_chain(embeddings)
                    st.session_state.docs_loaded = True
                st.success("PDFs loaded and indexed. Ask away!")

        # Status + tips
        st.markdown("---")
        st.caption("Status: " + ("Ready ✅" if st.session_state.docs_loaded else "Waiting for PDFs ⏳"))
        with st.expander("How to get great answers", expanded=False):
            st.markdown(
                "- Ask specific questions like Summarize section 3 or What are the key risks?\n"
                "- Reference page numbers if you know them.\n"
                "- You can upload more PDFs and re-process at any time."
            )

        
if __name__ == "__main__":
    main()