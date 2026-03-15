from pathlib import Path

from src.rag.repository import read_text_file
from src.utils.context_snippets import ContextSnippet, compose_context_snippets


def load_modified_file_examples(
    repo_path: str,
    modified_files: list[str],
    seen_samples: set[str],
) -> list[ContextSnippet]:
    repository_root = Path(repo_path).resolve()
    seen_paths: set[Path] = set()
    modified_file_examples: list[ContextSnippet] = []

    for relative_path in modified_files:
        file_path = (repository_root / relative_path).resolve()
        try:
            file_path.relative_to(repository_root)
        except ValueError:
            continue

        if file_path in seen_paths or not file_path.is_file():
            continue

        seen_paths.add(file_path)

        try:
            content = read_text_file(file_path).strip()
        except OSError:
            continue

        if not content or content in seen_samples:
            continue

        seen_samples.add(content)
        modified_file_examples.append(
            ContextSnippet(
                relative_path=relative_path,
                content=content,
                score=10_000,
                kind="modified",
            )
        )

    return modified_file_examples


def compose_related_files_context(
    file_examples: list[ContextSnippet],
) -> str:
    return compose_context_snippets(file_examples)
