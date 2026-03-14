from functools import lru_cache

from openai import OpenAI

from llm.settings import get_llm_api_key, get_llm_base_url, get_llm_model


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(
        api_key=get_llm_api_key(),
        base_url=get_llm_base_url(),
    )


def complete_text(prompt: str) -> str:
    response = get_client().chat.completions.create(
        model=get_llm_model(),
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The local LLM returned an empty completion.")

    return content
