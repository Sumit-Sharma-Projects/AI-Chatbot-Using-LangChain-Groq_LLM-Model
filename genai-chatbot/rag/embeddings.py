"""HuggingFace embeddings loader with process-level caching."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings_model(
    model_name: str = settings.embedding_model_name,
) -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace sentence-transformer embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
