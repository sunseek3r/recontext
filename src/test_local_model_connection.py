import os

from openai import OpenAI

from rag.settings import get_embedding_api_key

DEFAULT_TEST_INPUT = "connection test"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required to test the local model connection.")

    return value


def main() -> None:
    base_url = get_required_env("EMBEDDINGS_BASE_URL")
    model = get_required_env("EMBEDDINGS_MODEL")
    test_input = os.getenv("TEST_EMBEDDING_INPUT", DEFAULT_TEST_INPUT)

    client = OpenAI(
        api_key=get_embedding_api_key(),
        base_url=base_url,
    )
    response = client.embeddings.create(
        model=model,
        input=test_input,
    )
    embedding = response.data[0].embedding

    print("Connection OK")
    print(f"Model: {model}")
    print(f"Vector size: {len(embedding)}")


if __name__ == "__main__":
    main()
