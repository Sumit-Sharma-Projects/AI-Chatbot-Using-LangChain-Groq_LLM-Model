"""Local FAISS vector store management with persistent manifest tracking."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings
from rag.document_loader import DocumentProcessingError, load_document_from_path
from rag.embeddings import get_embeddings_model
from rag.text_splitter import split_documents


class VectorStoreManager:
    """Manages local FAISS indexing, persistence, manifest metadata, and re-indexing."""

    def __init__(self, store_dir: Optional[Path] = None, uploads_dir: Optional[Path] = None):
        self.store_dir = store_dir or settings.vectorstore_dir
        self.uploads_dir = uploads_dir or settings.uploads_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.store_dir / "manifest.json"
        self.index_file = self.store_dir / "index.faiss"

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"documents": {}, "total_chunks": 0, "updated_at": None}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"documents": {}, "total_chunks": 0, "updated_at": None}

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["total_chunks"] = sum(
            doc.get("chunks", 0) for doc in manifest.get("documents", {}).values()
        )
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def has_index(self) -> bool:
        """Return True if a persisted FAISS index exists on disk."""
        return self.index_file.exists()

    def load_vector_store(self) -> Optional[FAISS]:
        """Load the persisted FAISS vector store if present."""
        if not self.has_index():
            return None
        embeddings = get_embeddings_model()
        return FAISS.load_local(
            str(self.store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def get_index_summary(self) -> Dict[str, Any]:
        """Return structured metadata about all indexed documents and chunks."""
        manifest = self._read_manifest()
        docs = manifest.get("documents", {})
        return {
            "document_count": len(docs),
            "total_chunks": manifest.get("total_chunks", 0),
            "documents": [
                {"name": name, **meta} for name, meta in sorted(docs.items())
            ],
            "updated_at": manifest.get("updated_at"),
        }

    def index_file_path(self, file_path: Path) -> Dict[str, Any]:
        """Load, split, embed, and add a single file into the local FAISS store.

        If a file with the same name was previously indexed, the store is rebuilt
        cleanly so duplicate chunks never accumulate.
        """
        manifest = self._read_manifest()
        if file_path.name in manifest.get("documents", {}):
            return self.reindex_all()

        raw_docs = load_document_from_path(file_path)
        chunks = split_documents(raw_docs)
        if not chunks:
            raise DocumentProcessingError(
                f"Document '{file_path.name}' produced zero indexable chunks."
            )

        embeddings = get_embeddings_model()
        existing_store = self.load_vector_store()
        if existing_store is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            existing_store.add_documents(chunks)
            vectorstore = existing_store

        vectorstore.save_local(str(self.store_dir))

        manifest.setdefault("documents", {})[file_path.name] = {
            "pages_or_sections": len(raw_docs),
            "chunks": len(chunks),
            "file_type": file_path.suffix.lstrip(".").upper(),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(manifest)

        return {
            "file_name": file_path.name,
            "pages_or_sections": len(raw_docs),
            "chunks": len(chunks),
        }

    def reindex_all(self) -> Dict[str, Any]:
        """Rebuild the entire FAISS index from all files currently in data/uploads/."""
        self._delete_index_files()

        upload_files = sorted(
            path
            for path in self.uploads_dir.iterdir()
            if path.is_file() and path.suffix.lower() in settings.supported_extensions
        )

        if not upload_files:
            self._write_manifest({"documents": {}, "total_chunks": 0})
            return {"document_count": 0, "total_chunks": 0}

        all_chunks: List[Document] = []
        new_docs_meta: Dict[str, Dict[str, Any]] = {}

        for path in upload_files:
            raw_docs = load_document_from_path(path)
            chunks = split_documents(raw_docs)
            all_chunks.extend(chunks)
            new_docs_meta[path.name] = {
                "pages_or_sections": len(raw_docs),
                "chunks": len(chunks),
                "file_type": path.suffix.lstrip(".").upper(),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }

        if all_chunks:
            embeddings = get_embeddings_model()
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            vectorstore.save_local(str(self.store_dir))

        self._write_manifest({"documents": new_docs_meta})
        return {"document_count": len(new_docs_meta), "total_chunks": len(all_chunks)}

    def remove_document_and_reindex(self, filename: str) -> Dict[str, Any]:
        """Remove an uploaded document from disk and rebuild the FAISS index."""
        target = self.uploads_dir / filename
        if target.exists() and target.is_file():
            target.unlink()
        return self.reindex_all()

    def clear_all_documents(self) -> None:
        """Delete all uploaded documents and reset the local FAISS index."""
        for path in self.uploads_dir.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()
        self._delete_index_files()
        self._write_manifest({"documents": {}, "total_chunks": 0})

    def _delete_index_files(self) -> None:
        for file_name in ("index.faiss", "index.pkl"):
            candidate = self.store_dir / file_name
            if candidate.exists():
                candidate.unlink()
