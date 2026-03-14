import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from src.config import REPO_PATH
from src.regex_search.repository import (
    FILE_GLOBS,
    NAME_GLOBS,
    SKIP_DIRECTORY_GLOBS,
    iter_searchable_repository_files,
    read_text_file,
)
from src.regex_search.settings import get_regex_chunk_line_limit


@lru_cache(maxsize=None)
def get_cached_file_content(path: str) -> str:
    return read_text_file(Path(path))


def build_match_document(
    *,
    repo_path: Path,
    file_path: Path,
    content: str,
    match_line_number: int,
    chunk_line_limit: int,
) -> Document | None:
    lines = content.splitlines()
    if not lines:
        return None

    lines_before_match = (chunk_line_limit - 1) // 2
    line_start = max(match_line_number - lines_before_match, 1)
    line_end = min(line_start + chunk_line_limit - 1, len(lines))
    line_start = max(line_end - chunk_line_limit + 1, 1)
    snippet = "\n".join(lines[line_start - 1 : line_end]).strip()
    if not snippet:
        return None

    return Document(
        page_content=snippet,
        metadata={
            "source": str(file_path),
            "relative_path": str(file_path.relative_to(repo_path)),
            "match_line": match_line_number,
            "line_start": line_start,
            "line_end": line_end,
        },
    )


def get_document_key(document: Document) -> tuple[str, int, int]:
    return (
        document.metadata["relative_path"],
        document.metadata["line_start"],
        document.metadata["line_end"],
    )


def build_ripgrep_command(pattern: str, repo_path: Path) -> list[str]:
    command = [
        "rg",
        "--json",
        "--line-number",
        "--hidden",
        "--no-ignore-vcs",
        "--color",
        "never",
        pattern,
        str(repo_path),
    ]

    for glob in FILE_GLOBS:
        command.extend(["--glob", glob])

    for glob in NAME_GLOBS:
        command.extend(["--glob", glob])

    for glob in SKIP_DIRECTORY_GLOBS:
        command.extend(["--glob", glob])

    return command


def search_with_ripgrep(
    pattern: str,
    repo_path: Path,
    limit: int,
    chunk_line_limit: int,
) -> list[Document]:
    completed_process = subprocess.run(
        build_ripgrep_command(pattern, repo_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode not in {0, 1}:
        raise RuntimeError(completed_process.stderr.strip() or "ripgrep search failed.")

    documents: list[Document] = []
    seen_documents: set[tuple[str, int, int]] = set()

    for line in completed_process.stdout.splitlines():
        event = json.loads(line)
        if event["type"] != "match":
            continue

        file_path = Path(event["data"]["path"]["text"])
        if not file_path.is_absolute():
            file_path = repo_path / file_path
        file_path = file_path.resolve()
        content = get_cached_file_content(str(file_path))
        document = build_match_document(
            repo_path=repo_path,
            file_path=file_path,
            content=content,
            match_line_number=event["data"]["line_number"],
            chunk_line_limit=chunk_line_limit,
        )
        if document is None:
            continue

        document_key = get_document_key(document)
        if document_key in seen_documents:
            continue

        seen_documents.add(document_key)
        documents.append(document)
        if len(documents) >= limit:
            break

    return documents


def search_with_python_regex(
    pattern: str,
    repo_path: Path,
    limit: int,
    chunk_line_limit: int,
) -> list[Document]:
    compiled_pattern = re.compile(pattern, re.MULTILINE)
    documents: list[Document] = []
    seen_documents: set[tuple[str, int, int]] = set()

    for file_path in iter_searchable_repository_files(repo_path):
        content = read_text_file(file_path)
        for match in compiled_pattern.finditer(content):
            match_line_number = content.count("\n", 0, match.start()) + 1
            document = build_match_document(
                repo_path=repo_path,
                file_path=file_path,
                content=content,
                match_line_number=match_line_number,
                chunk_line_limit=chunk_line_limit,
            )
            if document is None:
                continue

            document_key = get_document_key(document)
            if document_key in seen_documents:
                continue

            seen_documents.add(document_key)
            documents.append(document)
            if len(documents) >= limit:
                return documents

    return documents


def regex_search(
    pattern: str,
    repo_path: str | Path = REPO_PATH,
    limit: int = 4,
) -> list[Document]:
    if limit <= 0:
        return []

    normalized_repo_path = Path(repo_path).resolve()
    chunk_line_limit = get_regex_chunk_line_limit()

    if shutil.which("rg"):
        return search_with_ripgrep(
            pattern,
            normalized_repo_path,
            limit,
            chunk_line_limit,
        )

    return search_with_python_regex(
        pattern,
        normalized_repo_path,
        limit,
        chunk_line_limit,
    )
