"""Retrieval-Augmented Generation (RAG) package for document indexing and retrieval."""

from rag.document_loader import DocumentProcessingError, load_document_from_path, save_uploaded_file
from rag.embeddings import get_embeddings_model
from rag.retriever import RAGRetriever, SourceReference
from rag.text_splitter import split_documents
from rag.vector_store import VectorStoreManager

__all__ = [
    "DocumentProcessingError",
    "RAGRetriever",
    "SourceReference",
    "VectorStoreManager",
    "get_embeddings_model",
    "load_document_from_path",
    "save_uploaded_file",
    "split_documents",
]
