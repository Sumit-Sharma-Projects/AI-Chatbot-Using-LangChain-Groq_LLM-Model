"""Modern LangChain orchestrator supporting both Direct Streaming RAG and AI Agent mode."""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from agents.tools import build_agent_tools
from config.settings import settings
from llm.groq_model import GroqConfigurationError, classify_groq_error, get_chat_groq
from memory.chat_history import ChatSessionManager
from rag.retriever import RAGRetriever, SourceReference
from utils.streaming import stream_langchain_chain


@dataclass
class AgentExecutionResult:
    """Structured result returned by the AI Agent execution path."""

    answer: str
    tools_used: List[str] = field(default_factory=list)
    sources: List[SourceReference] = field(default_factory=list)
    error: Optional[str] = None


class ChatOrchestrator:
    """Coordinates Direct Streaming LLM+RAG and LangChain Tool-Calling Agent modes."""

    def __init__(
        self,
        session_manager: ChatSessionManager,
        retriever: Optional[RAGRetriever] = None,
    ):
        self.session_manager = session_manager
        self.retriever = retriever or RAGRetriever()

    def stream_direct_response(
        self,
        session_id: str,
        user_query: str,
        model_name: str,
        temperature: float,
        rag_enabled: bool,
        top_k: int = settings.default_top_k,
    ) -> Tuple[Generator[str, None, None], List[SourceReference]]:
        """Stream a token-by-token response using RunnableWithMessageHistory and optional RAG."""
        sources: List[SourceReference] = []
        rag_context_block = "No external documents were queried for this turn."

        if rag_enabled and self.retriever.store_manager.has_index():
            context_text, sources = self.retriever.retrieve(query=user_query, top_k=top_k)
            if context_text:
                rag_context_block = (
                    "Retrieved Document Context:\n"
                    f"{context_text}\n\n"
                    "Grounding Rules:\n"
                    "- When answering questions about the uploaded documents, rely strictly on the "
                    "retrieved excerpts above and mention document names and page numbers.\n"
                    "- If the user asks about something not covered in the uploaded documents, "
                    "explicitly state that the uploaded documents do not contain that information "
                    "instead of inventing details."
                )
            else:
                rag_context_block = (
                    "RAG is enabled and documents are indexed, but similarity search returned no "
                    "matching excerpts for this query. If the user is asking about their documents, "
                    "state clearly that the information was not found in the uploaded files."
                )

        llm = get_chat_groq(
            model_name=model_name,
            temperature=temperature,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a knowledgeable, helpful, text-only AI QnA assistant. "
                    "Respond clearly using well-structured Markdown and formatted code blocks when useful.\n\n"
                    "{rag_context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm

        chain_with_history = RunnableWithMessageHistory(
            chain,
            self.session_manager.get_langchain_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        raw_stream = chain_with_history.stream(
            {"input": user_query, "rag_context": rag_context_block},
            config={"configurable": {"session_id": session_id}},
        )
        return stream_langchain_chain(raw_stream), sources

    def run_agent_response(
        self,
        session_id: str,
        user_query: str,
        model_name: str,
        temperature: float,
        rag_enabled: bool,
        top_k: int = settings.default_top_k,
    ) -> AgentExecutionResult:
        """Execute a modern LangChain tool-calling agent with isolated session history."""
        collected_sources: List[SourceReference] = []

        try:
            llm = get_chat_groq(
                model_name=model_name,
                temperature=temperature,
                streaming=False,
            )

            tools = build_agent_tools(
                model_name=model_name,
                temperature=temperature,
                rag_enabled=rag_enabled,
                top_k=top_k,
                retriever=self.retriever,
                collected_sources=collected_sources,
            )

            rag_status_note = (
                "Uploaded documents are indexed and searchable via document_search_tool."
                if rag_enabled and self.retriever.store_manager.has_index()
                else (
                    "RAG is enabled but no documents are uploaded yet."
                    if rag_enabled
                    else "RAG document search is currently disabled."
                )
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an intelligent, text-only AI QnA Agent.\n"
                        "Choose tools deliberately and avoid unnecessary tool calls:\n"
                        "- For simple greetings, follow-ups, or straightforward questions, answer directly "
                        "or use general_llm_tool only if helpful.\n"
                        "- Use document_search_tool whenever the user asks about uploaded files, PDFs, "
                        "reports, spreadsheets, or specific document content.\n"
                        "- Use calculator_tool for mathematical calculations.\n"
                        "- Use knowledge_retrieval_tool for reference definitions on RAG, Groq, LangChain, "
                        "Embeddings, FAISS, or Streamlit.\n"
                        "Current Document Index Status: {rag_status}\n"
                        "Always respond in clear Markdown. Never fabricate document facts: if a document "
                        "search yields no matching answer, clearly inform the user.",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=5,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )

            agent_with_history = RunnableWithMessageHistory(
                executor,
                self.session_manager.get_langchain_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            result: Dict[str, Any] = agent_with_history.invoke(
                {"input": user_query, "rag_status": rag_status_note},
                config={"configurable": {"session_id": session_id}},
            )

            tools_used: List[str] = []
            for action, _observation in result.get("intermediate_steps", []):
                tool_name = getattr(action, "tool", None)
                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)

            # Deduplicate collected sources while preserving order
            deduped_sources: List[SourceReference] = []
            seen = set()
            for src in collected_sources:
                key = (src.source_file, src.page_number, src.snippet[:60])
                if key not in seen:
                    seen.add(key)
                    deduped_sources.append(src)

            output_text = str(result.get("output") or "").strip()
            if not output_text:
                output_text = "I could not generate a response. Please rephrase your question."

            return AgentExecutionResult(
                answer=output_text,
                tools_used=tools_used,
                sources=deduped_sources,
            )

        except GroqConfigurationError as exc:
            return AgentExecutionResult(answer="", error=str(exc))
        except Exception as exc:
            return AgentExecutionResult(answer="", error=classify_groq_error(exc))
