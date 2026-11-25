"""LangChain tools for general QnA, RAG document search, safe math, and knowledge retrieval."""

import ast
import math
import operator
from typing import Any, Callable, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from config.settings import settings
from llm.groq_model import classify_groq_error, get_chat_groq
from rag.retriever import RAGRetriever, SourceReference

# Safe AST operators for calculator_tool
_BINARY_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MATH_FUNCS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pow": pow,
}

_MATH_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

# Built-in curated knowledge base for Knowledge Retrieval Tool
_LOCAL_KNOWLEDGE_BASE: Dict[str, str] = {
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines semantic vector search with LLM "
        "generation. Documents are loaded, split into overlapping chunks, embedded into "
        "dense vectors (e.g., all-MiniLM-L6-v2), stored in a vector index such as FAISS, "
        "and retrieved by similarity to ground model answers with verifiable citations."
    ),
    "groq": (
        "Groq provides ultra-low-latency LLM inference powered by Language Processing Units "
        "(LPUs). Through langchain-groq's ChatGroq integration, applications gain fast "
        "token streaming and native tool-calling support for Llama, Mixtral, and Gemma models."
    ),
    "langchain": (
        "Modern LangChain uses LCEL (LangChain Expression Language), RunnableWithMessageHistory "
        "with InMemoryChatMessageHistory for isolated conversation sessions, and "
        "create_tool_calling_agent + AgentExecutor for structured multi-tool reasoning."
    ),
    "embeddings": (
        "Sentence-transformer embeddings (such as sentence-transformers/all-MiniLM-L6-v2) map "
        "text chunks into 384-dimensional unit vectors so cosine/L2 similarity search retrieves "
        "semantically related passages even when exact keywords differ."
    ),
    "faiss": (
        "FAISS (Facebook AI Similarity Search) is a local vector similarity library that stores "
        "dense embeddings on disk (index.faiss and index.pkl) without requiring an external "
        "database server."
    ),
    "streamlit": (
        "Streamlit provides reactive Python web interfaces with st.session_state, st.chat_input, "
        "st.chat_message, and st.write_stream for real-time ChatGPT-style conversational apps."
    ),
}


def _safe_eval_ast(node: ast.AST) -> float:
    """Safely evaluate a parsed Python arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in _MATH_CONSTS:
            return _MATH_CONSTS[node.id]
        raise ValueError(f"Unsupported symbol '{node.id}'.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return float(_BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _safe_eval_ast(node.operand)
        return float(_UNARY_OPS[type(node.op)](operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _MATH_FUNCS:
            raise ValueError(f"Unsupported function '{func_name}'.")
        args = [_safe_eval_ast(arg) for arg in node.args]
        return float(_MATH_FUNCS[func_name](*args))
    raise ValueError("Unsupported mathematical expression.")


def build_agent_tools(
    model_name: str,
    temperature: float,
    rag_enabled: bool,
    top_k: int,
    retriever: RAGRetriever,
    collected_sources: List[SourceReference],
) -> List[Any]:
    """Build request-scoped LangChain tools bound to runtime settings and RAG state."""

    @tool
    def general_llm_tool(question: str) -> str:
        """Answer general knowledge questions or conversational prompts that do not require uploaded documents or math calculations."""
        try:
            llm = get_chat_groq(
                model_name=model_name,
                temperature=temperature,
                streaming=False,
            )
            response = llm.invoke(
                [
                    SystemMessage(
                        content="Provide a clear, accurate, concise text response."
                    ),
                    HumanMessage(content=question),
                ]
            )
            return str(response.content)
        except Exception as exc:
            return f"General LLM tool error: {classify_groq_error(exc)}"

    @tool
    def document_search_tool(query: str) -> str:
        """Search the user's uploaded documents (PDF, TXT, DOCX, CSV) for specific facts, policies, data, or passages."""
        if not rag_enabled:
            return (
                "Document search is currently turned OFF in Application Settings. "
                "Inform the user they can enable RAG in the sidebar to search uploaded files."
            )

        if not retriever.store_manager.has_index():
            return (
                "No documents have been uploaded or indexed yet. "
                "Let the user know they can upload PDF, TXT, DOCX, or CSV files in the sidebar."
            )

        try:
            context, references = retriever.retrieve(query=query, top_k=top_k)
            if not context:
                return (
                    "No relevant excerpts matching this query were found in the uploaded documents. "
                    "Do not fabricate an answer; clearly tell the user the uploaded documents "
                    "do not contain this information."
                )

            collected_sources.extend(references)
            citations = ", ".join(ref.format_citation() for ref in references)
            return (
                f"Retrieved Document Excerpts (Sources: {citations}):\n\n{context}\n\n"
                "Use ONLY these excerpts to answer document-specific questions and cite the "
                "document name and page number whenever available. If the answer is not "
                "present in the excerpts, state that clearly."
            )
        except Exception as exc:
            return f"Document search failed: {exc}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Evaluate arithmetic and mathematical expressions accurately (e.g., '24 * (15 + 3)', 'sqrt(144)', 'round(100 / 3, 2)')."""
        try:
            cleaned = (expression or "").strip().replace("^", "**")
            if not cleaned:
                return "Calculator error: Empty expression."
            tree = ast.parse(cleaned, mode="eval")
            result = _safe_eval_ast(tree)
            if result.is_integer():
                return str(int(result))
            return f"{result:.6g}"
        except ZeroDivisionError:
            return "Calculator error: Division by zero is undefined."
        except Exception as exc:
            return f"Calculator error: Could not evaluate '{expression}' ({exc})."

    @tool
    def knowledge_retrieval_tool(topic: str) -> str:
        """Retrieve curated technical reference knowledge about RAG, Groq, LangChain, Embeddings, FAISS, or Streamlit."""
        try:
            query = (topic or "").strip().lower()
            if not query:
                return "Knowledge retrieval error: Provide a topic keyword."

            matches = [
                f"[{key.upper()}] {summary}"
                for key, summary in _LOCAL_KNOWLEDGE_BASE.items()
                if key in query or any(word in summary.lower() for word in query.split())
            ]
            if matches:
                return "\n\n".join(matches)

            available = ", ".join(sorted(_LOCAL_KNOWLEDGE_BASE.keys()))
            return (
                f"No exact entry found in the curated local knowledge base for '{topic}'. "
                f"Available topics: {available}. Try general_llm_tool or document_search_tool."
            )
        except Exception as exc:
            return f"Knowledge retrieval error: {exc}"

    tools = [
        document_search_tool,
        calculator_tool,
        knowledge_retrieval_tool,
        general_llm_tool,
    ]
    return tools
