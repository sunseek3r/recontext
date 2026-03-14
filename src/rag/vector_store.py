import os
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import REPO_PATH
from rag.embeddings import get_embedding_configuration_key, get_embeddings
from rag.repository import load_code_repository

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHROMA_DIR_PATH = "chroma"


def get_persist_directory(repo_path: str | Path, embedding_key: str) -> str:
    repository_name = os.path.basename(os.path.realpath(repo_path))
    return os.path.join(CHROMA_DIR_PATH, repository_name, embedding_key)


def get_collection_name(repo_path: str | Path, embedding_key: str) -> str:
    return f"{Path(repo_path).resolve().name}-{embedding_key}"


def index_documents(vector_store: Chroma, repo_path: str | Path) -> None:
    documents = load_code_repository(repo_path)
    if not documents:
        raise RuntimeError(f"No indexable documents found in {repo_path}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    chunk_ids = [
        f"{chunk.metadata['relative_path']}:{index}" for index, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(documents=chunks, ids=chunk_ids)


def is_vector_store_empty(vector_store: Chroma) -> bool:
    result = vector_store.get(limit=1, include=[])
    return not result["ids"]


@lru_cache(maxsize=None)
def get_cached_vector_store(repo_path: str, embedding_key: str) -> Chroma:
    persist_directory = get_persist_directory(repo_path, embedding_key)
    os.makedirs(persist_directory, exist_ok=True)

    vector_store = Chroma(
        collection_name=get_collection_name(repo_path, embedding_key),
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )

    if is_vector_store_empty(vector_store):
        index_documents(vector_store, repo_path)

    return vector_store


def build_vector_store(repo_path: str | Path) -> Chroma:
    normalized_repo_path = os.path.realpath(repo_path)
    return get_cached_vector_store(
        normalized_repo_path,
        get_embedding_configuration_key(),
    )


def similarity_search(query: str, repo_path: str | Path = REPO_PATH, limit: int = 4):
    vector_store = build_vector_store(repo_path)
    return vector_store.similarity_search(query, k=limit)
