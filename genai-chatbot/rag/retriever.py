"""Similarity-based RAG retriever with structured source citations."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from config.settings import settings
from rag.vector_store import VectorStoreManager


@dataclass
class SourceReference:
    """Metadata and excerpt for a retrieved document chunk."""

    source_file: str
    page_number: Optional[int]
    score: float
    snippet: str

    def format_citation(self) -> str:
        """Format human-readable source reference with page number if available."""
        if self.page_number is not None:
            return f"{self.source_file} (Page {self.page_number})"
        return self.source_file


class RAGRetriever:
    """Retrieves relevant context chunks and citations from the local FAISS store."""

    def __init__(self, store_manager: Optional[VectorStoreManager] = None):
        self.store_manager = store_manager or VectorStoreManager()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.default_top_k,
    ) -> Tuple[str, List[SourceReference]]:
        """Run similarity search for a query and return formatted context + citations.

        Returns:
            A tuple of (formatted_context_block, list_of_SourceReference).
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return "", []

        vectorstore = self.store_manager.load_vector_store()
        if vectorstore is None:
            return "", []

        results = vectorstore.similarity_search_with_score(cleaned_query, k=top_k)
        if not results:
            return "", []

        context_parts: List[str] = []
        references: List[SourceReference] = []
        seen_keys = set()

        for idx, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata or {}
            source_file = str(meta.get("source_file", "Unknown Document"))
            page_num = meta.get("page_number")
            excerpt = " ".join((doc.page_content or "").split())
            snippet = excerpt[:280] + ("..." if len(excerpt) > 280 else "")

            page_label = f" | Page {page_num}" if page_num is not None else ""
            context_parts.append(
                f"[Source {idx}: {source_file}{page_label}]\n{doc.page_content}"
            )

            dedupe_key = (source_file, page_num, snippet[:80])
            if dedupe_key not in seen_keys:
                seen_keys.add(dedupe_key)
                references.append(
                    SourceReference(
                        source_file=source_file,
                        page_number=page_num,
                        score=float(score),
                        snippet=snippet,
                    )
                )

        return "\n\n---\n\n".join(context_parts), references
