import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_RAG_QUERY = "logic for user authentication"
ALLOWED_EMBEDDING_MODEL_PRESETS = {"openai", "self-hosted-openai"}


def get_rag_query() -> str:
    return os.getenv("RAG_QUERY", DEFAULT_RAG_QUERY)


def get_embedding_model_preset() -> str:
    preset = os.getenv("EMBEDDING_MODEL_PRESET", "openai")
    if preset not in ALLOWED_EMBEDDING_MODEL_PRESETS:
        raise RuntimeError(
            "EMBEDDING_MODEL_PRESET must be one of: openai, self-hosted-openai."
        )

    return preset


def get_embedding_api_key() -> str:
    api_key = os.getenv("EMBEDDINGS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set EMBEDDINGS_API_KEY or OPENAI_API_KEY to build the RAG vector store."
        )

    return api_key
