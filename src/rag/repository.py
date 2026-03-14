import os
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

MAX_FILE_SIZE_BYTES = 1_000_000
SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
TEXT_FILE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_FILE_NAMES = {
    ".env.example",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README",
    "README.md",
    "requirements.txt",
}


def is_indexable_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size <= MAX_FILE_SIZE_BYTES
        and (path.suffix.lower() in TEXT_FILE_EXTENSIONS or path.name in TEXT_FILE_NAMES)
    )


def iter_repository_files(repo_path: Path) -> Iterator[Path]:
    for root, dir_names, file_names in os.walk(repo_path):
        dir_names[:] = sorted(
            directory for directory in dir_names if directory not in SKIP_DIRECTORIES
        )
        current_dir = Path(root)
        for file_name in sorted(file_names):
            file_path = current_dir / file_name
            if is_indexable_file(file_path):
                yield file_path


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def load_code_repository(repo_path: str | Path) -> list[Document]:
    repository_root = Path(repo_path).resolve()
    documents: list[Document] = []

    for file_path in iter_repository_files(repository_root):
        content = read_text_file(file_path).strip()
        if not content:
            continue

        relative_path = file_path.relative_to(repository_root)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "relative_path": str(relative_path),
                },
            )
        )

    return documents
