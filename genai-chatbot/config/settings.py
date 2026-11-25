"""Centralized application configuration and environment management.

Loads secrets securely via python-dotenv and exposes validated settings
for the LLM, RAG pipeline, AI agents, and local persistence directories.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Resolve project root directory and load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class AppSettings:
    """Immutable runtime settings for the chatbot application."""

    app_name: str = "NexusQnA Assistant"
    app_subtitle: str = "Text-Based Conversational QnA with Groq, LangChain Agents & RAG"

    # Directory paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    chat_sessions_dir: Path = BASE_DIR / "data" / "chat_sessions"
    logo_path: Path = BASE_DIR / "assets" / "logo.png"

    # Available Groq models (Model ID -> Friendly Display Label)
    groq_models: Dict[str, str] = field(
        default_factory=lambda: {
            "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Balanced & Capable)",
            "llama-3.1-8b-instant": "Llama 3.1 8B Instant (Ultra-Fast)",
            "mixtral-8x7b-32768": "Mixtral 8x7B 32K (Long Context)",
            "gemma2-9b-it": "Gemma 2 9B IT (Compact & Crisp)",
        }
    )

    default_model: str = os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    max_tokens: int = 2048

    # RAG & Embedding configuration
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 4

    # Upload validation constraints
    supported_extensions: Tuple[str, ...] = (".pdf", ".txt", ".docx", ".csv")
    max_upload_size_mb: int = 25

    def ensure_directories(self) -> None:
        """Create required local storage directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.uploads_dir,
            self.vectorstore_dir,
            self.chat_sessions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_groq_api_key() -> str:
        """Return the Groq API key from environment variables (never hardcoded)."""
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
        return os.getenv("GROQ_API_KEY", "").strip()

    @classmethod
    def validate_groq_api_key(cls) -> Tuple[bool, str]:
        """Validate presence and basic shape of the Groq API key."""
        api_key = cls.get_groq_api_key()
        if not api_key:
            return (
                False,
                "GROQ_API_KEY is missing. Add it to your .env file (see .env.example).",
            )
        if api_key in {"your_groq_api_key_here", "MY_GROQ_API_KEY", "gsk_..."}:
            return (
                False,
                "GROQ_API_KEY is still set to a placeholder value. Replace it in .env.",
            )
        if not api_key.startswith("gsk_"):
            return (
                False,
                "GROQ_API_KEY appears invalid. Groq API keys typically start with 'gsk_'.",
            )
        return True, "Groq API credentials configured."

    def get_model_ids(self) -> List[str]:
        """Return ordered list of supported Groq model IDs."""
        return list(self.groq_models.keys())


settings = AppSettings()
settings.ensure_directories()
