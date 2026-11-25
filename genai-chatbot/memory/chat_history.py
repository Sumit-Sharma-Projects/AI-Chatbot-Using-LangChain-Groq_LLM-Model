"""Modern LangChain message history & local JSON session persistence.

Uses langchain_core.chat_history.InMemoryChatMessageHistory keyed by unique
session_id so RunnableWithMessageHistory maintains strict per-session isolation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings


class ChatSessionManager:
    """Manages multi-session chat persistence and isolated LangChain histories."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or settings.chat_sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lc_store: Dict[str, InMemoryChatMessageHistory] = {}

    def _session_path(self, session_id: str) -> Path:
        safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
        return self.sessions_dir / f"{safe_id}.json"

    def create_session(self, title: str = "New Chat") -> str:
        """Create and persist a brand-new chat session with a unique UUID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._write_session_file(session_id, payload)
        self._lc_store[session_id] = InMemoryChatMessageHistory()
        return session_id

    def _write_session_file(self, session_id: str, payload: Dict[str, Any]) -> None:
        self._session_path(session_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session dictionary from disk by unique ID."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all saved sessions ordered from most recently updated to oldest."""
        sessions: List[Dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "id": data.get("id", path.stem),
                        "title": data.get("title", "Untitled Chat"),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                continue

        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Return persisted message dicts for UI rendering."""
        data = self.load_session(session_id)
        if not data:
            return []
        return data.get("messages", [])

    def get_langchain_history(self, session_id: str) -> BaseChatMessageHistory:
        """Return an isolated InMemoryChatMessageHistory for RunnableWithMessageHistory."""
        if session_id in self._lc_store:
            return self._lc_store[session_id]

        history = InMemoryChatMessageHistory()
        for msg in self.get_messages(session_id):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                history.add_message(HumanMessage(content=content))
            elif role == "assistant":
                history.add_message(AIMessage(content=content))

        self._lc_store[session_id] = history
        return history

    def sync_from_session_state(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Synchronize in-memory LangChain history cleanly from stored messages."""
        history = InMemoryChatMessageHistory()
        for msg in messages:
            if msg.get("role") == "user":
                history.add_message(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                history.add_message(AIMessage(content=msg.get("content", "")))
        self._lc_store[session_id] = history

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        tools_used: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Persist a single message to the session JSON file and update title if first prompt."""
        data = self.load_session(session_id)
        if data is None:
            self.create_session()
            data = self.load_session(session_id) or {
                "id": session_id,
                "title": "New Chat",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
            }

        now = datetime.now(timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": now,
        }
        if sources:
            entry["sources"] = sources
        if tools_used:
            entry["tools_used"] = tools_used

        messages = data.setdefault("messages", [])
        messages.append(entry)

        # Auto-title new conversations from the first user prompt
        if role == "user" and (data.get("title") in {"New Chat", "", None}):
            cleaned = " ".join(content.split())
            data["title"] = (cleaned[:44] + "...") if len(cleaned) > 44 else cleaned

        data["updated_at"] = now
        self._write_session_file(session_id, data)
        return entry

    def clear_session_messages(self, session_id: str) -> None:
        """Remove all messages inside a session while keeping the session ID active."""
        data = self.load_session(session_id)
        if data is None:
            return
        data["messages"] = []
        data["title"] = "New Chat"
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_session_file(session_id, data)
        self._lc_store[session_id] = InMemoryChatMessageHistory()

    def delete_session(self, session_id: str) -> None:
        """Delete a conversation session permanently from disk and memory."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        self._lc_store.pop(session_id, None)

    def clear_all_sessions(self) -> None:
        """Delete all saved conversation sessions."""
        for path in self.sessions_dir.glob("*.json"):
            path.unlink()
        self._lc_store.clear()
