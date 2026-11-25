"""Input validation, formatting, and chat export utilities."""

from datetime import datetime
import json
from typing import Any, Dict, List, Tuple


def validate_user_input(prompt: str) -> Tuple[bool, str]:
    """Ensure user prompt is non-empty and within a reasonable text size."""
    if prompt is None:
        return False, "Please enter a question or message."
    cleaned = prompt.strip()
    if not cleaned:
        return False, "Empty messages cannot be sent. Please type your question."
    if len(cleaned) > 16000:
        return False, "Message exceeds 16,000 characters. Please shorten your prompt."
    return True, cleaned


def format_timestamp_short(iso_str: str) -> str:
    """Format an ISO-8601 timestamp into a concise local display string."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return ""


def export_chat_as_markdown(title: str, messages: List[Dict[str, Any]]) -> str:
    """Serialize a conversation session into clean, copy-ready Markdown."""
    lines = [f"# {title or 'Conversation Export'}", ""]
    for msg in messages:
        role_label = "User" if msg.get("role") == "user" else "Assistant"
        timestamp = format_timestamp_short(msg.get("timestamp", ""))
        header = f"### {role_label}" + (f" ({timestamp})" if timestamp else "")
        lines.append(header)
        lines.append(msg.get("content", ""))

        tools = msg.get("tools_used") or []
        if tools:
            lines.append(f"\n*Tools used:* {', '.join(tools)}")

        sources = msg.get("sources") or []
        if sources:
            lines.append("\n*Sources:*")
            for src in sources:
                file_name = src.get("source_file", "Document")
                page = src.get("page_number")
                loc = f" (Page {page})" if page is not None else ""
                lines.append(f"- {file_name}{loc}")
        lines.append("\n---\n")
    return "\n".join(lines)


def export_chat_as_json(
    session_id: str, title: str, messages: List[Dict[str, Any]]
) -> str:
    """Serialize a conversation session into formatted JSON."""
    payload = {
        "session_id": session_id,
        "title": title,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
