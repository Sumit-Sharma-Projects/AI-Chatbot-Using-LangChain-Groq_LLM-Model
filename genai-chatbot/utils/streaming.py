"""Real-time response streaming utilities compatible with st.write_stream()."""

import time
from typing import Any, Generator, Iterable

from llm.groq_model import classify_groq_error


def extract_token_text(chunk: Any) -> str:
    """Extract clean string text from diverse LangChain stream chunk shapes."""
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
    if isinstance(chunk, dict):
        if "output" in chunk and isinstance(chunk["output"], str):
            return chunk["output"]
        if "answer" in chunk and isinstance(chunk["answer"], str):
            return chunk["answer"]
        if "content" in chunk and isinstance(chunk["content"], str):
            return chunk["content"]
    return ""


def stream_langchain_chain(
    stream_iterable: Iterable[Any],
    on_error_prefix: str = "\n\n[Stream interrupted] ",
) -> Generator[str, None, None]:
    """Yield text tokens from a LangChain .stream() generator with graceful error recovery."""
    emitted = False
    try:
        for chunk in stream_iterable:
            token = extract_token_text(chunk)
            if token:
                emitted = True
                yield token
    except Exception as exc:
        friendly = classify_groq_error(exc)
        if emitted:
            yield f"{on_error_prefix}{friendly}"
        else:
            yield f"Unable to complete response: {friendly}"


def stream_text_smoothly(text: str, words_per_step: int = 3, delay_sec: float = 0.018) -> Generator[str, None, None]:
    """Stream finalized agent text progressively for a ChatGPT-like visual cadence."""
    if not text:
        return
    tokens = text.split(" ")
    for idx in range(0, len(tokens), words_per_step):
        segment = " ".join(tokens[idx : idx + words_per_step])
        if idx + words_per_step < len(tokens):
            segment += " "
        yield segment
        time.sleep(delay_sec)
