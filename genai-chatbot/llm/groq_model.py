"""Modular Groq LLM initialization and error classification using langchain-groq."""

from typing import Optional

from langchain_groq import ChatGroq

from config.settings import settings


class GroqConfigurationError(Exception):
    """Raised when Groq API credentials or model parameters are invalid."""


class GroqAPIError(Exception):
    """Raised when a Groq inference request fails at runtime."""


def get_chat_groq(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = True,
) -> ChatGroq:
    """Create and return a configured ChatGroq instance.

    Args:
        model_name: Groq model identifier (defaults to settings.default_model).
        temperature: Sampling temperature between 0.0 and 1.0.
        max_tokens: Maximum output tokens per response.
        streaming: Enable token-by-token streaming responses.

    Returns:
        Configured ChatGroq client ready for LangChain chains or agents.

    Raises:
        GroqConfigurationError: If GROQ_API_KEY is missing or invalid.
    """
    is_valid, message = settings.validate_groq_api_key()
    if not is_valid:
        raise GroqConfigurationError(message)

    selected_model = model_name or settings.default_model
    selected_temp = (
        settings.default_temperature if temperature is None else float(temperature)
    )
    selected_temp = max(0.0, min(1.0, selected_temp))
    selected_max_tokens = max_tokens or settings.max_tokens

    try:
        return ChatGroq(
            api_key=settings.get_groq_api_key(),
            model=selected_model,
            temperature=selected_temp,
            max_tokens=selected_max_tokens,
            streaming=streaming,
            max_retries=2,
        )
    except Exception as exc:
        raise GroqConfigurationError(
            f"Failed to initialize ChatGroq ({selected_model}): {classify_groq_error(exc)}"
        ) from exc


def classify_groq_error(exc: Exception) -> str:
    """Translate raw Groq / HTTP exceptions into safe, actionable user messages."""
    raw = str(exc)
    lower = raw.lower()

    if "401" in lower or "invalid_api_key" in lower or "authentication" in lower:
        return (
            "Authentication failed with Groq API. Please verify your GROQ_API_KEY in .env."
        )
    if "429" in lower or "rate limit" in lower or "rate_limit" in lower:
        return (
            "Groq API rate limit reached. Wait a few seconds or switch to a faster model "
            "such as llama-3.1-8b-instant."
        )
    if "model_not_found" in lower or "decommissioned" in lower or "404" in lower:
        return (
            "The selected Groq model is unavailable or deprecated. Please pick another model "
            "from the sidebar."
        )
    if "timeout" in lower or "connection" in lower or "network" in lower:
        return (
            "Network error while contacting Groq API. Check your internet connection and retry."
        )
    if "context_length" in lower or "too many tokens" in lower:
        return (
            "The conversation or retrieved context exceeded the model's token limit. "
            "Try starting a new chat or lowering RAG Top-K."
        )

    return f"Unexpected LLM error: {raw}"
