import hashlib
import os

from langchain_openai import OpenAIEmbeddings

from rag.settings import get_embedding_api_key, get_embedding_model_preset


def get_openai_embeddings_config() -> dict[str, object]:
    config: dict[str, object] = {
        "api_key": get_embedding_api_key(),
    }
    model = os.getenv("EMBEDDINGS_MODEL")
    organization = os.getenv("EMBEDDINGS_ORGANIZATION")
    dimensions = os.getenv("EMBEDDINGS_DIMENSIONS")

    if model:
        config["model"] = model
    if organization:
        config["organization"] = organization
    if dimensions:
        config["dimensions"] = int(dimensions)

    return config


def get_self_hosted_openai_embeddings_config() -> dict[str, object]:
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "EMBEDDINGS_BASE_URL is required when EMBEDDING_MODEL_PRESET=self-hosted-openai."
        )

    config = get_openai_embeddings_config()
    config["base_url"] = base_url

    return config


def get_embedding_configuration_key() -> str:
    raw_key = "|".join(
        [
            get_embedding_model_preset(),
            os.getenv("EMBEDDINGS_MODEL", ""),
            os.getenv("EMBEDDINGS_BASE_URL", ""),
            os.getenv("EMBEDDINGS_ORGANIZATION", ""),
            os.getenv("EMBEDDINGS_DIMENSIONS", ""),
        ]
    )
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]


def get_embeddings() -> OpenAIEmbeddings:
    preset = get_embedding_model_preset()
    if preset == "self-hosted-openai":
        config = get_self_hosted_openai_embeddings_config()
    else:
        config = get_openai_embeddings_config()

    return OpenAIEmbeddings(**config)
