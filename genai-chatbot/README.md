# NexusQnA Assistant — Text-Based Conversational QnA Chatbot

A modular, text-only conversational QnA web application built with **Python**, **Streamlit**, **LangChain**, **Groq LLM**, **HuggingFace Embeddings**, **FAISS**, **RunnableWithMessageHistory**, and **LangChain Tool-Calling Agents**.

## Features

- **ChatGPT-Style Text Interface**: Right-aligned user messages, left-aligned AI responses, fixed bottom chat input, Markdown + code formatting, and live streaming.
- **Groq LLM Integration**: Fast inference via `langchain-groq` (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`) with configurable temperature.
- **Modern Conversational Memory**: Isolated per-session memory powered by `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`, persisted locally in `data/chat_sessions/`.
- **Optional RAG Pipeline**: Upload `PDF`, `TXT`, `DOCX`, or `CSV` documents, chunk with `RecursiveCharacterTextSplitter`, embed with `sentence-transformers/all-MiniLM-L6-v2`, index in local **FAISS**, view source filenames/page numbers, and remove/re-index files anytime.
- **AI Agent Tool Routing**: Intelligent tool selection across `document_search_tool`, `calculator_tool`, `knowledge_retrieval_tool`, and `general_llm_tool` using `create_tool_calling_agent` + `AgentExecutor`.
- **Chat Export**: Download any conversation as Markdown (`.md`) or JSON (`.json`).

## Quick Start

```bash
cd genai-chatbot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY inside .env
streamlit run app.py
```
