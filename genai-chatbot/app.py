"""Main Streamlit web application for the Text-Based Conversational QnA Chatbot."""

from typing import Any, Dict, List

import streamlit as st

from agents.agent import ChatOrchestrator
from config.settings import settings
from llm.groq_model import GroqConfigurationError, classify_groq_error
from memory.chat_history import ChatSessionManager
from rag.document_loader import DocumentProcessingError, save_uploaded_file
from rag.retriever import RAGRetriever, SourceReference
from rag.vector_store import VectorStoreManager
from utils.helpers import (
    export_chat_as_json,
    export_chat_as_markdown,
    format_timestamp_short,
    validate_user_input,
)
from utils.streaming import stream_text_smoothly

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=settings.app_name,
    page_icon=str(settings.logo_path) if settings.logo_path.exists() else "💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Clean ChatGPT-inspired layout and typography */
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 5.5rem;
    max-width: 980px;
}

/* Right-aligned user message bubble */
.user-message-row {
    display: flex;
    justify-content: flex-end;
    margin: 0.65rem 0;
}

.user-bubble {
    background: #10a37f;
    color: #ffffff;
    padding: 0.8rem 1.05rem;
    border-radius: 1.1rem 1.1rem 0.25rem 1.1rem;
    max-width: 78%;
    font-size: 0.96rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(16, 163, 127, 0.16);
    word-wrap: break-word;
}

/* Left-aligned assistant container styling */
.assistant-meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 0.2rem;
}

.tool-tag {
    display: inline-block;
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 0.35rem;
    background: rgba(16, 163, 127, 0.12);
    color: #059669;
    margin-right: 0.35rem;
    margin-top: 0.25rem;
}

/* Subtle sidebar polish */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Service & Session State Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_services() -> Dict[str, Any]:
    """Initialize singleton backend managers once per Streamlit server process."""
    session_manager = ChatSessionManager()
    vector_manager = VectorStoreManager()
    retriever = RAGRetriever(store_manager=vector_manager)
    orchestrator = ChatOrchestrator(session_manager=session_manager, retriever=retriever)
    return {
        "session_manager": session_manager,
        "vector_manager": vector_manager,
        "retriever": retriever,
        "orchestrator": orchestrator,
    }


services = get_services()
session_manager: ChatSessionManager = services["session_manager"]
vector_manager: VectorStoreManager = services["vector_manager"]
orchestrator: ChatOrchestrator = services["orchestrator"]


def init_session_state() -> None:
    """Initialize Streamlit session state variables safely."""
    if "active_session_id" not in st.session_state:
        existing_sessions = session_manager.list_sessions()
        if existing_sessions:
            st.session_state.active_session_id = existing_sessions[0]["id"]
        else:
            st.session_state.active_session_id = session_manager.create_session()

    if "messages" not in st.session_state:
        st.session_state.messages = session_manager.get_messages(
            st.session_state.active_session_id
        )
        session_manager.sync_from_session_state(
            st.session_state.active_session_id, st.session_state.messages
        )

    st.session_state.setdefault("selected_model", settings.default_model)
    st.session_state.setdefault("temperature", settings.default_temperature)
    st.session_state.setdefault("rag_enabled", True)
    st.session_state.setdefault("agent_enabled", True)
    st.session_state.setdefault("top_k", settings.default_top_k)
    st.session_state.setdefault("last_tools_used", [])


def switch_session(session_id: str) -> None:
    """Switch active conversation and load its isolated history."""
    st.session_state.active_session_id = session_id
    st.session_state.messages = session_manager.get_messages(session_id)
    session_manager.sync_from_session_state(session_id, st.session_state.messages)
    st.session_state.last_tools_used = []


def start_new_chat() -> None:
    """Create a fresh conversation session and activate it."""
    new_id = session_manager.create_session()
    switch_session(new_id)


init_session_state()


# -----------------------------------------------------------------------------
# Helper Rendering Functions
# -----------------------------------------------------------------------------
def render_sources_expander(sources: List[Dict[str, Any]]) -> None:
    """Display retrieved document citations and excerpts in a collapsible panel."""
    if not sources:
        return
    with st.expander(f"Source References ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, start=1):
            file_name = src.get("source_file", "Document")
            page = src.get("page_number")
            location = f" — Page {page}" if page is not None else ""
            snippet = src.get("snippet", "")
            st.markdown(f"**{idx}. {file_name}{location}**")
            if snippet:
                st.caption(snippet)


def render_message(msg: Dict[str, Any]) -> None:
    """Render user messages on the right and AI responses on the left."""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    if role == "user":
        safe_html = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        st.markdown(
            f'<div class="user-message-row"><div class="user-bubble">{safe_html}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        col_left, _col_spacer = st.columns([0.86, 0.14])
        with col_left:
            with st.chat_message("assistant"):
                st.markdown(content)
                tools_used = msg.get("tools_used") or []
                if tools_used:
                    badges = "".join(
                        f'<span class="tool-tag">Tool: {tool_name}</span>'
                        for tool_name in tools_used
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                render_sources_expander(msg.get("sources") or [])


def serialize_sources(refs: List[SourceReference]) -> List[Dict[str, Any]]:
    """Convert SourceReference objects into JSON-serializable dicts."""
    return [
        {
            "source_file": ref.source_file,
            "page_number": ref.page_number,
            "score": round(ref.score, 4),
            "snippet": ref.snippet,
        }
        for ref in refs
    ]


# -----------------------------------------------------------------------------
# Sidebar (All 10 Required Sections)
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Application Logo and Name
    logo_col, title_col = st.columns([0.24, 0.76])
    with logo_col:
        if settings.logo_path.exists():
            st.image(str(settings.logo_path), width=48)
    with title_col:
        st.markdown(f"### {settings.app_name}")
        st.caption("Groq · LangChain · RAG")

    # 2. New Chat
    if st.button("＋ New Chat", type="primary", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()

    # 3. Chat History
    st.subheader("Chat History")
    all_sessions = session_manager.list_sessions()
    if not all_sessions:
        st.caption("No saved conversations yet.")
    else:
        for sess in all_sessions:
            sess_id = sess["id"]
            is_active = sess_id == st.session_state.active_session_id
            label_prefix = "▶ " if is_active else ""
            title_text = sess["title"] or "New Chat"
            short_time = format_timestamp_short(sess.get("updated_at", ""))
            btn_label = f"{label_prefix}{title_text}"

            c_open, c_del = st.columns([0.82, 0.18])
            with c_open:
                if st.button(
                    btn_label,
                    key=f"open_{sess_id}",
                    help=f"Session ID: {sess_id} | {short_time}",
                    use_container_width=True,
                ):
                    switch_session(sess_id)
                    st.rerun()
            with c_del:
                if st.button("✕", key=f"del_{sess_id}", help="Delete conversation"):
                    session_manager.delete_session(sess_id)
                    if is_active:
                        remaining = session_manager.list_sessions()
                        if remaining:
                            switch_session(remaining[0]["id"])
                        else:
                            start_new_chat()
                    st.rerun()

    st.divider()

    # 4. LLM Model Selection
    st.subheader("LLM Model Selection")
    model_ids = settings.get_model_ids()
    current_idx = (
        model_ids.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_ids
        else 0
    )
    st.session_state.selected_model = st.selectbox(
        "Groq Model",
        options=model_ids,
        index=current_idx,
        format_func=lambda m: settings.groq_models.get(m, m),
    )

    # 5. Temperature Configuration
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
        help="Lower values produce more factual answers; higher values increase creativity.",
    )

    st.divider()

    # 6. RAG Document Upload Section
    st.subheader("RAG Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or CSV",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
        help=f"Maximum {settings.max_upload_size_mb} MB per file.",
    )

    if uploaded_files and st.button("Process & Index Documents", use_container_width=True):
        with st.spinner("Extracting text, chunking, and building FAISS embeddings..."):
            success_count = 0
            for uf in uploaded_files:
                try:
                    saved_path = save_uploaded_file(uf)
                    stats = vector_manager.index_file_path(saved_path)
                    success_count += 1
                    st.toast(
                        f"Indexed {stats.get('file_name', uf.name)} "
                        f"({stats.get('chunks', 0)} chunks)"
                    )
                except DocumentProcessingError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unexpected error processing '{uf.name}': {exc}")

            if success_count > 0:
                st.success(f"Indexed {success_count} file(s) successfully.")
                st.rerun()

    # 7. Uploaded Document Information
    st.subheader("Uploaded Document Info")
    index_summary = vector_manager.get_index_summary()
    st.caption(
        f"Documents: **{index_summary['document_count']}** · "
        f"Chunks: **{index_summary['total_chunks']}**"
    )

    if index_summary["documents"]:
        for doc_meta in index_summary["documents"]:
            d_col, r_col = st.columns([0.82, 0.18])
            with d_col:
                st.caption(
                    f"📄 **{doc_meta['name']}** "
                    f"({doc_meta.get('file_type', '')}, {doc_meta.get('chunks', 0)} chunks)"
                )
            with r_col:
                if st.button("🗑", key=f"rm_doc_{doc_meta['name']}", help="Remove & re-index"):
                    with st.spinner(f"Removing {doc_meta['name']} and re-indexing..."):
                        vector_manager.remove_document_and_reindex(doc_meta["name"])
                    st.rerun()

        if st.button("Clear All Indexed Documents", use_container_width=True):
            vector_manager.clear_all_documents()
            st.rerun()
    else:
        st.caption("No documents indexed. Chatbot works normally without documents.")

    st.divider()

    # 8. AI Agent Status
    st.subheader("AI Agent Status")
    if st.session_state.agent_enabled:
        st.success("Agent Mode: Active (Multi-Tool Routing)")
        st.caption(
            "Available tools: Document Search, Calculator, Knowledge Retrieval, General LLM"
        )
    else:
        st.info("Direct LLM Mode: Active (Token Streaming)")

    if st.session_state.last_tools_used:
        st.caption(
            "Last tools invoked: " + ", ".join(st.session_state.last_tools_used)
        )

    st.divider()

    # 9. Application Settings
    st.subheader("Application Settings")
    st.session_state.rag_enabled = st.toggle(
        "Enable RAG (Document Search)",
        value=bool(st.session_state.rag_enabled),
    )
    st.session_state.agent_enabled = st.toggle(
        "Enable AI Agent Mode",
        value=bool(st.session_state.agent_enabled),
    )
    st.session_state.top_k = st.slider(
        "RAG Top-K Chunks",
        min_value=1,
        max_value=8,
        value=int(st.session_state.top_k),
        disabled=not st.session_state.rag_enabled,
    )

    # Export current conversation
    active_sess = session_manager.load_session(st.session_state.active_session_id) or {}
    active_title = active_sess.get("title", "Conversation")
    if st.session_state.messages:
        md_export = export_chat_as_markdown(active_title, st.session_state.messages)
        json_export = export_chat_as_json(
            st.session_state.active_session_id,
            active_title,
            st.session_state.messages,
        )
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                "Export MD",
                data=md_export,
                file_name=f"chat_{st.session_state.active_session_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "Export JSON",
                data=json_export,
                file_name=f"chat_{st.session_state.active_session_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()

    # 10. Clear Chat History
    st.subheader("Clear Chat History")
    if st.button("Clear Current Conversation", use_container_width=True):
        session_manager.clear_session_messages(st.session_state.active_session_id)
        st.session_state.messages = []
        st.session_state.last_tools_used = []
        st.rerun()

    if st.button("Delete All Chat History", use_container_width=True):
        session_manager.clear_all_sessions()
        start_new_chat()
        st.rerun()


# -----------------------------------------------------------------------------
# Main Chat Header & Credential Validation
# -----------------------------------------------------------------------------
header_left, header_btn1, header_btn2 = st.columns([0.66, 0.17, 0.17])
with header_left:
    st.markdown(f"## {settings.app_name}")
    rag_label = "RAG ON" if st.session_state.rag_enabled else "RAG OFF"
    agent_label = "Agent ON" if st.session_state.agent_enabled else "Direct Stream"
    st.caption(
        f"{settings.app_subtitle} · `{st.session_state.selected_model}` · "
        f"{rag_label} · {agent_label}"
    )
with header_btn1:
    if st.button("New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()
with header_btn2:
    if st.button("Clear Chat", use_container_width=True):
        session_manager.clear_session_messages(st.session_state.active_session_id)
        st.session_state.messages = []
        st.session_state.last_tools_used = []
        st.rerun()

creds_ok, creds_msg = settings.validate_groq_api_key()
if not creds_ok:
    st.warning(creds_msg)

st.divider()

# -----------------------------------------------------------------------------
# Conversation Feed
# -----------------------------------------------------------------------------
if not st.session_state.messages:
    st.info(
        "Ask any question below to start a conversation. Optionally upload PDF, TXT, DOCX, "
        "or CSV files in the sidebar to query your own documents with citations."
    )
else:
    for message in st.session_state.messages:
        render_message(message)


# -----------------------------------------------------------------------------
# Fixed Bottom Text Input & Streaming Generation
# -----------------------------------------------------------------------------
user_prompt = st.chat_input("Message NexusQnA Assistant...")

if user_prompt is not None:
    is_valid, cleaned_or_error = validate_user_input(user_prompt)
    if not is_valid:
        st.warning(cleaned_or_error)
    else:
        # 1. Persist and render user message on the right
        user_entry = session_manager.append_message(
            session_id=st.session_state.active_session_id,
            role="user",
            content=cleaned_or_error,
        )
        st.session_state.messages.append(user_entry)
        render_message(user_entry)

        # 2. Generate and stream AI response on the left
        col_left, _col_spacer = st.columns([0.86, 0.14])
        with col_left:
            with st.chat_message("assistant"):
                try:
                    if st.session_state.agent_enabled:
                        with st.spinner("Thinking and selecting tools..."):
                            agent_res = orchestrator.run_agent_response(
                                session_id=st.session_state.active_session_id,
                                user_query=cleaned_or_error,
                                model_name=st.session_state.selected_model,
                                temperature=float(st.session_state.temperature),
                                rag_enabled=bool(st.session_state.rag_enabled),
                                top_k=int(st.session_state.top_k),
                            )

                        if agent_res.error:
                            st.error(agent_res.error)
                            assistant_text = f"Error: {agent_res.error}"
                            tools_used: List[str] = []
                            serialized_sources: List[Dict[str, Any]] = []
                        else:
                            assistant_text = st.write_stream(
                                stream_text_smoothly(agent_res.answer)
                            )
                            tools_used = agent_res.tools_used
                            serialized_sources = serialize_sources(agent_res.sources)
                            st.session_state.last_tools_used = tools_used

                            if tools_used:
                                badges = "".join(
                                    f'<span class="tool-tag">Tool: {t}</span>'
                                    for t in tools_used
                                )
                                st.markdown(badges, unsafe_allow_html=True)
                            render_sources_expander(serialized_sources)
                    else:
                        with st.spinner("Generating response..."):
                            token_stream, refs = orchestrator.stream_direct_response(
                                session_id=st.session_state.active_session_id,
                                user_query=cleaned_or_error,
                                model_name=st.session_state.selected_model,
                                temperature=float(st.session_state.temperature),
                                rag_enabled=bool(st.session_state.rag_enabled),
                                top_k=int(st.session_state.top_k),
                            )
                        assistant_text = st.write_stream(token_stream)
                        tools_used = []
                        serialized_sources = serialize_sources(refs)
                        st.session_state.last_tools_used = []
                        render_sources_expander(serialized_sources)

                    # 3. Persist assistant response in session history
                    assistant_entry = session_manager.append_message(
                        session_id=st.session_state.active_session_id,
                        role="assistant",
                        content=str(assistant_text),
                        sources=serialized_sources,
                        tools_used=tools_used,
                    )
                    st.session_state.messages.append(assistant_entry)

                except GroqConfigurationError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(classify_groq_error(exc))
