import os

from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required to use the local LLM.")

    return value


def get_llm_base_url() -> str:
    return get_required_env("LLM_BASE_URL")


def get_llm_model() -> str:
    return get_required_env("LLM_MODEL")


def get_llm_api_key() -> str:
    return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
