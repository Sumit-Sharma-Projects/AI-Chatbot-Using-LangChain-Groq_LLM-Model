"""Document validation, persistence, and LangChain Document Loader routing."""

from pathlib import Path
import re
from typing import Any, List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from config.settings import settings


class DocumentProcessingError(Exception):
    """Raised when an uploaded document fails validation or text extraction."""


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename preserving the original extension."""
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem).strip("._") or "document"
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}"


def validate_uploaded_file(uploaded_file: Any) -> str:
    """Validate uploaded file extension, non-empty payload, and size limit."""
    if uploaded_file is None or not getattr(uploaded_file, "name", None):
        raise DocumentProcessingError("No file was provided for upload.")

    safe_name = sanitize_filename(uploaded_file.name)
    ext = Path(safe_name).suffix.lower()

    if ext not in settings.supported_extensions:
        supported = ", ".join(settings.supported_extensions)
        raise DocumentProcessingError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}."
        )

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise DocumentProcessingError(f"Uploaded file '{safe_name}' is empty.")

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise DocumentProcessingError(
            f"File '{safe_name}' ({size_mb:.1f} MB) exceeds the "
            f"{settings.max_upload_size_mb} MB limit."
        )

    return safe_name


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Validate and write a Streamlit UploadedFile to data/uploads/."""
    safe_name = validate_uploaded_file(uploaded_file)
    destination = settings.uploads_dir / safe_name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def load_document_from_path(file_path: Path) -> List[Document]:
    """Load a supported document from disk using LangChain document loaders.

    Supports PDF, TXT, DOCX, and CSV while normalizing metadata for citations.
    """
    if not file_path.exists():
        raise DocumentProcessingError(f"File not found: {file_path.name}")

    ext = file_path.suffix.lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext == ".csv":
            loader = CSVLoader(str(file_path), encoding="utf-8")
        else:
            raise DocumentProcessingError(f"Unsupported file format: {ext}")

        documents = loader.load()
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to extract text from '{file_path.name}': {exc}"
        ) from exc

    cleaned_docs: List[Document] = []
    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        metadata = dict(doc.metadata or {})
        metadata["source_file"] = file_path.name
        metadata["file_type"] = ext.lstrip(".")

        # Normalize 0-based PyPDFLoader page index into 1-based human page number
        if "page" in metadata and isinstance(metadata["page"], int):
            metadata["page_number"] = metadata["page"] + 1

        cleaned_docs.append(Document(page_content=text, metadata=metadata))

    if not cleaned_docs:
        raise DocumentProcessingError(
            f"No readable text was found in '{file_path.name}'."
        )

    return cleaned_docs
