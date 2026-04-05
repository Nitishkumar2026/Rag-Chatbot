"""
app.py — Streamlit RAG Chatbot with Real-Time Streaming
DataSphere Technologies Inc. — AI Assignment

Run:
    streamlit run app.py
"""

import sys
import os
import time
import streamlit as st

# Suppress HuggingFace transformers warnings (e.g. UNEXPECTED embeddings.position_ids)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DataSphere RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/yourusername/rag-chatbot",
        "Report a bug": None,
        "About": "Fine-Tuned RAG Chatbot | Amlgo Labs Assignment",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp { background: #0d0f17; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111320 0%, #0d0f17 100%);
        border-right: 1px solid #1e2235;
    }

    /* ── Chat message bubbles ── */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px 8px;
        margin-bottom: 4px;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #1a1f3a 0%, #252b45 100%);
        border-left: 3px solid #7c3aed;
        color: #e2e8f0 !important;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: linear-gradient(135deg, #0f1629 0%, #141d30 100%);
        border-left: 3px solid #06b6d4;
        color: #e2e8f0 !important;
    }

    /* ── Input box ── */
    [data-testid="stChatInputTextArea"] {
        background: #151823 !important;
        border: 1px solid #2a2f4a !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        color: #c7d2fe;
        border: 1px solid #4338ca;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #312e81, #4f46e5);
        border-color: #6366f1;
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    /* ── Metric cards in sidebar ── */
    [data-testid="stMetric"] {
        background: #141828;
        border: 1px solid #1e2540;
        border-radius: 10px;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem; }
    [data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 1.2rem; font-weight: 600; }

    /* ── Source expander ── */
    [data-testid="stExpander"] {
        background: #10141f;
        border: 1px solid #1e2540;
        border-radius: 10px;
    }

    /* ── Divider ── */
    hr { border-color: #1e2235; }

    /* ── Header gradient text ── */
    .hero-title {
        background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 4px;
    }

    /* ── Source chunk cards ── */
    .source-card {
        background: #0f1520;
        border: 1px solid #1d2540;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .source-card-header {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    /* ── Status badge ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f17; }
    ::-webkit-scrollbar-thumb { background: #1e2540; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3561; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline Loader (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the RAG pipeline once and cache across sessions."""
    sys.path.insert(0, str(Path(__file__).parent))
    from src.pipeline import RAGPipeline
    from src.document_processor import extract_pdf_text
    return RAGPipeline(top_k=3), extract_pdf_text


# ── Session State Init ────────────────────────────────────────────────────────

def init_session():
    """Initialise session-state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list of {role, content, sources}
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "temp_retriever" not in st.session_state:
        st.session_state.temp_retriever = None
    if "source_mode" not in st.session_state:
        st.session_state.source_mode = "Global KB"


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_source_cards(sources: list[dict]) -> None:
    """Render retrieved source chunks as styled cards inside an expander."""
    if not sources:
        return

    with st.expander(f"📚 **View {len(sources)} Source Chunk(s) Used**", expanded=False):
        for chunk in sources:
            score = chunk.get("similarity_score", 0)
            rank  = chunk.get("rank", "?")
            text  = chunk.get("text", "")
            words = chunk.get("word_count", len(text.split()))

            pct = int(score * 100)
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-card-header">
                        Source {rank} &nbsp;·&nbsp;
                        Relevance: {pct}% &nbsp;·&nbsp;
                        {words} words
                    </div>
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar(pipeline, extract_pdf_func) -> None:
    """Render the information sidebar."""
    with st.sidebar:
        # ── Branding ─────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 16px 0 8px;">
                <div style="font-size:2.5rem;">🤖</div>
                <div style="color:#a5b4fc; font-weight:700; font-size:1.1rem;">DataSphere RAG</div>
                <div style="color:#475569; font-size:0.75rem;">Powered by Local LLM + FAISS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Pipeline Stats ────────────────────────────────────────────────────
        if pipeline:
            stats = pipeline.get_stats()
            st.markdown(
                '<span class="status-dot"></span>'
                '<span style="color:#22c55e; font-size:0.8rem; font-weight:600;">PIPELINE ACTIVE</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📦 Chunks", f"{stats['chunk_count']:,}")
            with col2:
                st.metric("🔢 Top-K", stats["top_k"])

            st.markdown(
                f"""
                <div style="background:#141828; border:1px solid #1e2540; border-radius:10px;
                            padding:10px 14px; margin-top:8px;">
                    <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:0.05em; margin-bottom:4px;">Active Model</div>
                    <div style="color:#a5b4fc; font-size:0.88rem; font-weight:600;">
                        {stats['model']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ── PDF Upload ────────────────────────────────────────────────────────
        st.markdown(
            '<div style="color:#64748b; font-size:0.78rem; text-transform:uppercase; '
            'letter-spacing:0.05em; margin-bottom:8px;">📁 Upload Document</div>',
            unsafe_allow_html=True,
        )
        
        uploaded_file = st.file_uploader(
            "Add a PDF to chat with it",
            type=["pdf"],
            help="The file will be indexed in-memory and available for the current session.",
            label_visibility="collapsed"
        )

        if uploaded_file:
            file_id = f"pdf_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_id") != file_id:
                with st.spinner("📑 Indexing PDF …"):
                    try:
                        raw_text = extract_pdf_func(uploaded_file)
                        st.session_state.temp_retriever = pipeline.build_temp_retriever(raw_text)
                        st.session_state.last_uploaded_id = file_id
                        st.success(f"Indexed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error parsing PDF: {e}")

        # ── Source Toggle ─────────────────────────────────────────────────────
        if st.session_state.temp_retriever:
            st.markdown("")
            st.session_state.source_mode = st.radio(
                "Select Knowledge Source:",
                options=["Global KB", "Uploaded File"],
                index=1 if st.session_state.source_mode == "Uploaded File" else 0,
                help="Switch between the main document and your uploaded PDF."
            )

        st.divider()

        # ── Controls ──────────────────────────────────────────────────────────
        st.markdown(
            '<div style="color:#64748b; font-size:0.78rem; text-transform:uppercase; '
            'letter-spacing:0.05em; margin-bottom:8px;">Controls</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # ── Sample Queries ────────────────────────────────────────────────────
        st.markdown(
            '<div style="color:#64748b; font-size:0.78rem; text-transform:uppercase; '
            'letter-spacing:0.05em; margin-bottom:8px;">Sample Queries</div>',
            unsafe_allow_html=True,
        )

        sample_queries = [
            "What is the refund policy?",
            "How is user data encrypted?",
            "What are the rate limits for the Starter plan?",
            "What are the prohibited uses of the platform?",
            "How do I cancel my subscription?",
            "What compliance certifications does DataSphere hold?",
            "What personal data is collected?",
        ]

        for q in sample_queries:
            if st.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
                st.session_state["_inject_query"] = q
                st.rerun()

        st.divider()

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="color:#334155; font-size:0.72rem; text-align:center; padding-top:4px;">
                Amlgo Labs · Junior AI Engineer Assignment<br>
                FAISS · all-MiniLM-L6-v2 · LLM Backend
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    init_session()

    # ── Load Pipeline ─────────────────────────────────────────────────────────
    pipeline = None
    extract_pdf_func = None
    with st.spinner("⚙️  Loading RAG pipeline (first launch may take ~30s) …"):
        try:
            pipeline, extract_pdf_func = load_pipeline()
            st.session_state.pipeline_ready = True
        except Exception as e:
            st.session_state.pipeline_ready = False
            pipeline = None
            st.error(f"Error loading pipeline: {e}")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    if pipeline:
        render_sidebar(pipeline, extract_pdf_func)

    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown(
            '<h1 class="hero-title">DataSphere RAG Chatbot</h1>'
            '<p class="hero-sub">'
            'Ask anything about our Terms of Service, Privacy Policy &amp; more — '
            'answers are grounded in the document, not guessed.'
            '</p>',
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            """
            <div style="text-align:right; padding-top:12px;">
                <span style="background:#1e1b4b; color:#a5b4fc; padding:5px 12px;
                             border-radius:20px; font-size:0.75rem; border:1px solid #3730a3;">
                    🔍 RAG · Streaming
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Setup Error Banner ─────────────────────────────────────────────────────
    if not st.session_state.pipeline_ready:
        st.error(
            "**Pipeline not ready.** Run the setup steps first:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "python src/document_processor.py\n"
            "python src/embeddings.py\n"
            "```\n\n"
            "Then restart the app with `streamlit run app.py`.",
            icon="⚠️",
        )
        st.stop()

    # ── Render Chat History ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_source_cards(msg["sources"])

    # ── Welcome Message ────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0f1629,#141d30);
                        border:1px solid #1e2540; border-radius:14px;
                        padding:24px 28px; margin:16px 0;">
                <div style="color:#38bdf8; font-size:1.1rem; font-weight:600; margin-bottom:8px;">
                    👋 Welcome!
                </div>
                <div style="color:#94a3b8; line-height:1.8;">
                    I'm a RAG-powered chatbot trained on DataSphere's legal documentation.<br>
                    I can answer questions about:
                    <ul style="margin-top:8px; color:#64748b;">
                        <li>📋 Terms of Service &amp; account policies</li>
                        <li>🔒 Privacy Policy &amp; data handling</li>
                        <li>⚙️ API limits, pricing &amp; billing</li>
                        <li>🛡️ Security, compliance &amp; data processing</li>
                        <li>✅ Acceptable use &amp; refund policies</li>
                    </ul>
                    <span style="color:#475569; font-size:0.85rem;">
                        Try a sample query from the sidebar, or type your own question below.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Handle sidebar sample-query injection ──────────────────────────────────
    injected_query = st.session_state.pop("_inject_query", None)

    # ── Chat Input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask a question about the document…",
        disabled=not st.session_state.pipeline_ready,
    )

    # Use injected query if present, otherwise use typed input
    query = injected_query or user_input

    if query:
        # ── Display user message ───────────────────────────────────────────
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        st.session_state.messages.append({
            "role":    "user",
            "content": query,
            "sources": [],
        })

        # ── Generate & stream assistant response ───────────────────────────
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            sources_placeholder  = st.empty()

            # Show thinking indicator
            response_placeholder.markdown(
                f'<span style="color:#475569; font-style:italic; font-size:0.88rem;">'
                f'⏳ Searching {st.session_state.source_mode}…</span>',
                unsafe_allow_html=True,
            )

            try:
                # Select active retriever
                active_retriever = None
                if st.session_state.source_mode == "Uploaded File":
                    active_retriever = st.session_state.temp_retriever
                
                token_gen, sources = pipeline.run(query, retriever=active_retriever)

                # Stream tokens
                full_response = ""
                response_placeholder.markdown("")   # clear thinking indicator

                for token in token_gen:
                    full_response += token
                    # Update display with cursor
                    response_placeholder.markdown(
                        full_response + "▌",
                        unsafe_allow_html=False,
                    )

                # Final render (no cursor)
                response_placeholder.markdown(full_response)

                # Show sources below the answer
                with sources_placeholder.container():
                    render_source_cards(sources)

                # Persist to session
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as exc:
                err_msg = f"⚠️ **Error generating response:** {exc}"
                response_placeholder.error(err_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": err_msg,
                    "sources": [],
                })


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
