import streamlit as st
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.info-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

.answer-box {
    background-color: #111827;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #38bdf8;
    margin-top: 20px;
}

.chunk-box {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="title">📚 AI PDF Document Q&A</div>
    <div class="subtitle">
        Ask questions from your PDF documents using Groq + LangChain + FAISS
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.success("✅ Groq API Connected")

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.code("llama-3.1-8b-instant")

    st.markdown("### 📄 Embedding Model")
    st.code("all-MiniLM-L6-v2")

    st.markdown("### 📚 Vector Database")
    st.code("FAISS")

    st.markdown("---")

    st.info(
        "Click 'Create Vector DB' before asking questions."
    )


# =========================================================
# INITIALIZE LLM
# =========================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# =========================================================
# PROMPT TEMPLATE
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based only on the provided context.

Please provide the most accurate response.

<context>
{context}
</context>

Question: {input}
"""
)


# =========================================================
# VECTOR EMBEDDING FUNCTION
# =========================================================

def vector_embedding():

    if "vectors" not in st.session_state:

        with st.spinner("⚡ Creating Vector Database..."):

            # Embedding Model
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load PDF Documents
            st.session_state.loader = PyPDFDirectoryLoader("./your_data")

            st.session_state.docs = (
                st.session_state.loader.load()
            )

            # Split Documents
            st.session_state.text_splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            )

            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(
                    st.session_state.docs[:20]
                )
            )

            # Create FAISS Vector Store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )


# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([3, 1])

with col1:

    prompt1 = st.text_input(
        "💬 Ask a Question From Your Documents",
        placeholder="Example: What is the population growth mentioned in the report?"
    )

with col2:

    st.write("")
    st.write("")

    if st.button("📄 Create Vector DB"):
        vector_embedding()
        st.success("Vector Database Ready ✅")


# =========================================================
# QUESTION ANSWERING
# =========================================================

if prompt1:

    if "vectors" not in st.session_state:

        st.warning("⚠️ Please create the Vector Database first.")

    else:

        with st.spinner("🤖 Generating Response..."):

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            retriever = (
                st.session_state.vectors.as_retriever()
            )

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            start = time.process_time()

            response = retrieval_chain.invoke(
                {'input': prompt1}
            )

            response_time = (
                time.process_time() - start
            )

        # =================================================
        # ANSWER DISPLAY
        # =================================================

        st.markdown(
            f"""
            <div class="answer-box">
                <h3>🧠 Answer</h3>
                <p>{response['answer']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # METRICS
        # =================================================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="⏱️ Response Time",
                value=f"{response_time:.2f} sec"
            )

        with col2:
            st.metric(
                label="📄 Chunks Retrieved",
                value=len(response["context"])
            )

        with col3:
            st.metric(
                label="🤖 Model",
                value="Llama 3.1"
            )

        # =================================================
        # DOCUMENT CHUNKS
        # =================================================

        with st.expander("📚 Document Similarity Search"):

            for i, doc in enumerate(response["context"]):

                st.markdown(
                    f"""
                    <div class="chunk-box">
                        <h4>📄 Document Chunk {i+1}</h4>
                        <p>{doc.page_content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )