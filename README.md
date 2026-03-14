It is a Hackathon PoC project for collecting relevant repository context for fill-in-the-middle code completion tasks.

# Useful module methods

## `src/rag`

`src/rag/__init__.py` currently exposes one public helper:

- `similarity_search(query: str, repo_path: str | Path = REPO_PATH, limit: int = 4) -> list[Document]`

What it does:
- Builds or reuses a cached Chroma vector store for the target repository.
- Indexes text and code files when the store is empty.
- Returns the top `limit` LangChain `Document` matches for the query.

What each returned `Document` contains:
- `page_content`: matched code or text chunk.
- `metadata["source"]`: absolute source file path.
- `metadata["relative_path"]`: path relative to the indexed repository root.

Useful details:
- The default `repo_path` comes from `src/config.py` via `REPO_PATH`.
- The vector store is persisted under `chroma/<repo-name>/<embedding-config-hash>`.
- Supported embedding presets are `openai` and `self-hosted-openai`.

Required environment for `similarity_search`:
- `EMBEDDINGS_API_KEY` or `OPENAI_API_KEY`
- `EMBEDDING_MODEL_PRESET=self-hosted-openai` also requires `EMBEDDINGS_BASE_URL`
- Optional knobs: `EMBEDDINGS_MODEL`, `EMBEDDINGS_ORGANIZATION`, `EMBEDDINGS_DIMENSIONS`, `RAG_QUERY`

Example:

```python
from rag import similarity_search

results = similarity_search("function definition", limit=4)
top_result = results[0]
print(top_result.metadata["relative_path"])
print(top_result.page_content)
```

## `src/llm`

`src/llm/__init__.py` currently exposes one public helper:

- `complete_text(prompt: str) -> str`

What it does:
- Creates a cached OpenAI-compatible client.
- Sends the prompt as a single `user` chat message.
- Returns the first completion text.

Required environment for `complete_text`:
- `LLM_BASE_URL`
- `LLM_MODEL`
- Optional auth: `LLM_API_KEY` or `OPENAI_API_KEY`

Example:

```python
from llm import complete_text

result = complete_text("Write a short hello-world function in Python.")
print(result)
```
