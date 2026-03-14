from pathlib import Path
from typing import Iterator

from src.rag.repository import (
    SKIP_DIRECTORIES,
    TEXT_FILE_EXTENSIONS,
    TEXT_FILE_NAMES,
    iter_repository_files,
    read_text_file,
)

FILE_GLOBS = tuple(sorted({f"*{extension}" for extension in TEXT_FILE_EXTENSIONS}))
NAME_GLOBS = tuple(sorted(TEXT_FILE_NAMES))
SKIP_DIRECTORY_GLOBS = tuple(sorted(f"!**/{directory}/**" for directory in SKIP_DIRECTORIES))


def iter_searchable_repository_files(repo_path: str | Path) -> Iterator[Path]:
    yield from iter_repository_files(Path(repo_path).resolve())
