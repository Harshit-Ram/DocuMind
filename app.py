"""
DocMind - PDF Q&A Chatbot with RAG (Retrieval-Augmented Generation)
--------------------------------------------------------------------
Upload a PDF, ask questions, get answers grounded in that PDF.
Supports both Groq API and local TinyLlama model.

Run:
    py -3.12 -m pip install -r requirements.txt
    copy .env.example .env   (then put your GROQ_API_KEY inside)
    py -3.12 -m streamlit run app.py
"""

import hashlib
import html
import os

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------- CONFIG ----------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

st.set_page_config(page_title="DocMind - PDF Q&A", page_icon="🧠", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown(
    """
<style>
    .stApp { background: linear-gradient(180deg, #0f1117 0%, #161a23 100%); }
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #9ca3af; font-size: 0.95rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .status-pill {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc; font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem;
    }
    .chat-bubble-user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;
        padding: 12px 18px; border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 80%; margin-left: auto; font-size: 0.95rem;
        overflow-wrap: anywhere;
    }
    .chat-bubble-bot {
        background: #1f2430; border: 1px solid #2d3344; color: #e5e7eb;
        padding: 14px 18px; border-radius: 18px 18px 18px 4px;
        margin: 8px 0; max-width: 85%; font-size: 0.95rem; line-height: 1.5;
        overflow-wrap: anywhere;
    }
    .doc-card {
        background: #1a1f2e; border: 1px solid #2d3344; border-radius: 10px;
        padding: 16px; margin-bottom: 12px;
    }
    .doc-card-title { color: #e5e7eb; font-weight: 600; font-size: 0.95rem; }
    .doc-card-meta { color: #6b7280; font-size: 0.8rem; margin-top: 4px; }
    section[data-testid="stSidebar"] { background: #12141c; border-right: 1px solid #23283a; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- CACHED RESOURCES ----------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_local_model():
    from ctransformers import AutoModelForCausalLM
    if not os.path.exists(LOCAL_MODEL_PATH):
        return None
    return AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        model_type="llama",
        max_new_tokens=1024,
    )


def get_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def get_groq_client():
    from groq import Groq
    key = get_api_key()
    if not key:
        return None
    return Groq(api_key=key)


# ---------- CORE FUNCTIONS ----------
def file_hash(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()


def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    if chunk_size <= 0:
        return [text.strip()] if text.strip() else []
    overlap = max(0, min(overlap, chunk_size - 1))
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def build_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(
        chunks, show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(question, chunks, index, embed_model, k=DEFAULT_TOP_K):
    k = max(1, min(k, len(chunks)))
    q_emb = embed_model.encode(
        [question], normalize_embeddings=True
    ).astype("float32")
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def ask_groq(client, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document analysis assistant. Extract specific, detailed information "
                        "directly from the provided document context. Follow these rules:\n"
                        "- Always cite specific details, names, dates, numbers, and facts from the document.\n"
                        "- Quote or paraphrase exact text from the document to support your answer.\n"
                        "- Be thorough and comprehensive - cover all relevant points found in the context.\n"
                        "- Structure your answer clearly with bullet points or paragraphs as appropriate.\n"
                        "- If comparing or evaluating, reference specific sections of the document.\n"
                        "- If the answer is not in the context, say exactly: \"I couldn't find that in the document.\"\n"
                        "- Never give generic or vague responses. Always ground your answer in the actual document content."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context}\n\n"
                        f"QUESTION: {question}\n\n"
                        "Provide a detailed, specific answer based ONLY on the document above. "
                        "Include specific details, evidence, and quotes from the document."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}") from e
    return response.choices[0].message.content


def ask_local(llm, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a document analysis assistant. Answer the question using ONLY the document context below.\n"
        "Be specific. Cite details, names, dates, and facts from the document.\n"
        "If the answer is not in the context, say: \"I couldn't find that in the document.\"\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    try:
        output = llm(prompt, max_new_tokens=512, temperature=0.2)
        return output.strip() if output else "No response generated."
    except Exception as e:
        raise RuntimeError(f"Local model error: {e}") from e


# ---------- SESSION STATE ----------
for _key, _default in [
    ("chunks", None),
    ("index", None),
    ("pdf_name", None),
    ("pdf_hash", None),
    ("history", []),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


def clear_document():
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.pdf_name = None
    st.session_state.pdf_hash = None
    st.session_state.history = []


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🧠 DocMind")
    st.caption("RAG-powered document Q&A")
    st.divider()

    # Model mode selector
    local_model_available = os.path.exists(LOCAL_MODEL_PATH)
    if local_model_available and get_api_key():
        use_local = st.toggle("Use local model (TinyLlama)", value=False,
                              help="Toggle between Groq API and local TinyLlama model")
    elif local_model_available:
        use_local = True
        st.info("Using local TinyLlama model (no Groq API key set)")
    elif get_api_key():
        use_local = False
    else:
        st.warning("Set `GROQ_API_KEY` in `.env` or ensure local model exists.")
        use_local = False

    if use_local and not local_model_available:
        st.error("Local model not found at `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`")
        use_local = False

    if use_local:
        llm = load_local_model()
        if llm is None:
            st.error("Failed to load local model.")
    else:
        llm = None

    top_k = st.slider("Retrieval size (k)", 1, 8, DEFAULT_TOP_K)

    st.divider()
    st.markdown("#### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF to start asking questions about it.",
    )

    if uploaded_file is not None:
        digest = file_hash(uploaded_file)
        if st.session_state.pdf_hash != digest:
            embed_model = load_embed_model()
            with st.spinner("Processing document..."):
                try:
                    raw_text = extract_text_from_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"Could not read this PDF: {e}")
                    raw_text = ""
                if not raw_text.strip():
                    st.error(
                        "No extractable text found. This may be a scanned PDF "
                        "(images only). Try an OCR tool first."
                    )
                else:
                    chunks = chunk_text(raw_text)
                    if not chunks:
                        st.error("Could not split the document into chunks.")
                    else:
                        index = build_faiss_index(chunks, embed_model)
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.pdf_hash = digest
                        st.session_state.history = []

    if st.session_state.chunks:
        st.divider()
        model_label = "TinyLlama (local)" if use_local else GROQ_MODEL
        st.markdown(
            f"""<div class="doc-card">
                <div class="doc-card-title">📄 {html.escape(st.session_state.pdf_name)}</div>
                <div class="doc-card-meta">{len(st.session_state.chunks)} chunks indexed</div>
                <div class="doc-card-meta">Embedding: {EMBED_MODEL_NAME} (local)</div>
                <div class="doc-card-meta">LLM: {model_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Clear document", use_container_width=True):
            clear_document()
            st.rerun()


# ---------- MAIN AREA ----------
st.markdown('<p class="main-header">DocMind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about any PDF. Answers are retrieved from the actual document '
    "using embeddings + vector search, not guessed by the model.</p>",
    unsafe_allow_html=True,
)

if st.session_state.chunks is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

model_status = "TinyLlama (local)" if use_local else "Groq API"
st.markdown(
    f'<span class="status-pill">🟢 Document ready — {model_status} — ask anything</span>',
    unsafe_allow_html=True,
)

for q, a, _ in st.session_state.history:
    st.markdown(
        f'<div class="chat-bubble-user">{html.escape(q)}</div>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="chat-bubble-bot">{html.escape(a)}</div>', unsafe_allow_html=True
    )

question = st.chat_input("Ask a question about the document...")

if question:
    st.markdown(
        f'<div class="chat-bubble-user">{html.escape(question)}</div>',
        unsafe_allow_html=True,
    )

    embed_model = load_embed_model()

    with st.spinner("Retrieving relevant sections..."):
        relevant_chunks = retrieve_relevant_chunks(
            question,
            st.session_state.chunks,
            st.session_state.index,
            embed_model,
            k=top_k,
        )

    with st.spinner("Generating answer..."):
        try:
            if use_local:
                answer = ask_local(llm, question, relevant_chunks)
            else:
                groq_client = get_groq_client()
                if groq_client is None:
                    st.error(
                        "No Groq API key found. Set `GROQ_API_KEY=...` in your `.env` file "
                        "or enable local model in the sidebar."
                    )
                    st.stop()
                answer = ask_groq(groq_client, question, relevant_chunks)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.markdown(
        f'<div class="chat-bubble-bot">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.history.append((question, answer, relevant_chunks))
