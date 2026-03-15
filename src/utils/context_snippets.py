from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import keyword
from pathlib import Path
import re

from langchain_core.documents import Document

from src.rag.repository import iter_repository_files, read_text_file

FILE_SEPARATOR = "<|file_sep|>"
DEFAULT_SNIPPET_LINE_LIMIT = 20
MAX_QUERY_TERMS = 24
MAX_FILE_SNIPPETS = 2

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
SYMBOL_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+([A-Za-z0-9_,\s]+)|import\s+([A-Za-z0-9_,\s\.]+))",
    re.MULTILINE,
)
STRING_RE = re.compile(r"['\"]([A-Za-z_][A-Za-z0-9_./:-]{2,})['\"]")

COMMON_TERMS = {
    "args",
    "assert",
    "cls",
    "data",
    "dict",
    "false",
    "for",
    "from",
    "none",
    "pass",
    "return",
    "self",
    "str",
    "true",
    "tuple",
    "type",
    "value",
}


@dataclass(frozen=True)
class ContextSnippet:
    relative_path: str
    content: str
    score: int
    kind: str
    line_start: int | None = None
    line_end: int | None = None

    def render(self, *, file_separator: str = FILE_SEPARATOR) -> str:
        header = self.relative_path
        if self.line_start is not None and self.line_end is not None:
            header = f"{header}#L{self.line_start}-L{self.line_end}"

        return f"{file_separator}{header}\n{self.content.strip()}"


def document_to_snippet(
    document: Document,
    *,
    score: int,
    kind: str,
) -> ContextSnippet:
    return ContextSnippet(
        relative_path=document.metadata["relative_path"],
        content=document.page_content.strip(),
        score=score,
        kind=kind,
        line_start=document.metadata.get("line_start"),
        line_end=document.metadata.get("line_end"),
    )


def compose_context_snippets(
    snippets: list[ContextSnippet],
    *,
    file_separator: str = FILE_SEPARATOR,
) -> str:
    if not snippets:
        return ""

    return "".join(snippet.render(file_separator=file_separator) for snippet in snippets)


def collect_targeted_snippets(
    repo_path: str | Path,
    *,
    file_prefix: str,
    file_suffix: str,
    target_file_path: str | None,
    exclude_relative_paths: set[str] | None = None,
    limit: int = 8,
) -> list[ContextSnippet]:
    if limit <= 0:
        return []

    repository_root = Path(repo_path).resolve()
    excluded_paths = {
        (repository_root / relative_path).resolve()
        for relative_path in (exclude_relative_paths or set())
    }
    term_weights = extract_query_terms(
        file_prefix=file_prefix,
        file_suffix=file_suffix,
        target_file_path=target_file_path,
    )
    if not term_weights and target_file_path is None:
        return []

    snippets: list[ContextSnippet] = []
    for file_path in iter_repository_files(repository_root):
        resolved_file_path = file_path.resolve()
        if resolved_file_path in excluded_paths:
            continue

        content = read_text_file(resolved_file_path).strip()
        if not content:
            continue

        relative_path = str(resolved_file_path.relative_to(repository_root))
        path_score = score_path_match(
            relative_path=relative_path,
            target_file_path=target_file_path,
        )
        line_scores = score_content_lines(content, term_weights)
        if not line_scores and path_score <= 0:
            continue

        snippets.extend(
            build_snippets_for_file(
                relative_path=relative_path,
                content=content,
                line_scores=line_scores,
                path_score=path_score,
            )
        )

    deduped_snippets = deduplicate_snippets(snippets)
    deduped_snippets.sort(
        key=lambda snippet: (
            snippet.score,
            snippet.kind != "path",
            snippet.relative_path,
            snippet.line_start or 0,
        ),
        reverse=True,
    )
    return deduped_snippets[:limit]


def extract_query_terms(
    *,
    file_prefix: str,
    file_suffix: str,
    target_file_path: str | None,
) -> dict[str, int]:
    source = f"{file_prefix}\n{file_suffix}"
    weighted_terms: dict[str, int] = {}

    add_terms(weighted_terms, SYMBOL_DEF_RE.findall(source), weight=8)
    add_terms(weighted_terms, CALL_RE.findall(source), weight=5)
    add_terms(weighted_terms, extract_import_terms(source), weight=6)
    add_terms(weighted_terms, extract_string_terms(source), weight=4)

    identifier_counts = Counter(IDENTIFIER_RE.findall(source))
    for identifier, occurrences in identifier_counts.items():
        if not is_useful_term(identifier):
            continue

        weight = min(occurrences, 3) + 1
        if identifier[:1].isupper():
            weight += 2

        weighted_terms[identifier] = max(weighted_terms.get(identifier, 0), weight)

    if target_file_path is not None:
        add_terms(weighted_terms, extract_path_terms(target_file_path), weight=4)

    ranked_terms = sorted(
        weighted_terms.items(),
        key=lambda item: (item[1], len(item[0]), item[0]),
        reverse=True,
    )
    return dict(ranked_terms[:MAX_QUERY_TERMS])


def add_terms(weighted_terms: dict[str, int], terms: list[str], *, weight: int) -> None:
    for term in terms:
        if not is_useful_term(term):
            continue

        weighted_terms[term] = max(weighted_terms.get(term, 0), weight)


def extract_import_terms(source: str) -> list[str]:
    terms: list[str] = []
    for from_module, from_names, import_names in IMPORT_RE.findall(source):
        if from_module:
            terms.extend(split_symbol_terms(from_module))
            terms.extend(split_symbol_terms(from_names))
        elif import_names:
            terms.extend(split_symbol_terms(import_names))

    return terms


def extract_string_terms(source: str) -> list[str]:
    terms: list[str] = []
    for value in STRING_RE.findall(source):
        terms.extend(split_symbol_terms(value))

    return terms


def extract_path_terms(target_file_path: str) -> list[str]:
    path = Path(target_file_path)
    terms: list[str] = []
    for part in path.parts:
        terms.extend(split_symbol_terms(part))

    return terms


def split_symbol_terms(value: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9_]+", value)
    return [part for part in parts if part]


def is_useful_term(term: str) -> bool:
    normalized_term = term.strip()
    if len(normalized_term) < 3:
        return False

    if keyword.iskeyword(normalized_term):
        return False

    return normalized_term.lower() not in COMMON_TERMS


def score_path_match(
    *,
    relative_path: str,
    target_file_path: str | None,
) -> int:
    if target_file_path is None:
        return 0

    candidate_path = Path(relative_path)
    target_path = Path(target_file_path)
    score = 0

    if candidate_path.parent == target_path.parent:
        score += 8

    shared_parent_parts = 0
    for candidate_part, target_part in zip(candidate_path.parts[:-1], target_path.parts[:-1]):
        if candidate_part != target_part:
            break
        shared_parent_parts += 1

    score += min(shared_parent_parts, 3) * 2

    if candidate_path.stem == target_path.stem:
        score += 10

    if normalize_stem(candidate_path.stem) == normalize_stem(target_path.stem):
        score += 6

    candidate_name_tokens = set(split_symbol_terms(candidate_path.name))
    target_name_tokens = set(split_symbol_terms(target_path.name))
    score += min(len(candidate_name_tokens & target_name_tokens), 3) * 2

    return score


def normalize_stem(stem: str) -> str:
    normalized_stem = stem
    for prefix in ("test_", "tests_", "test"):
        if normalized_stem.startswith(prefix):
            normalized_stem = normalized_stem[len(prefix):]
            break

    for suffix in ("_test", "_tests"):
        if normalized_stem.endswith(suffix):
            normalized_stem = normalized_stem[: -len(suffix)]
            break

    return normalized_stem


def score_content_lines(content: str, term_weights: dict[str, int]) -> dict[int, int]:
    if not term_weights:
        return {}

    line_scores: dict[int, int] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        identifiers = set(IDENTIFIER_RE.findall(line))
        line_score = sum(term_weights[identifier] for identifier in identifiers if identifier in term_weights)
        if line_score > 0:
            line_scores[line_number] = line_score

    return line_scores


def build_snippets_for_file(
    *,
    relative_path: str,
    content: str,
    line_scores: dict[int, int],
    path_score: int,
) -> list[ContextSnippet]:
    lines = content.splitlines()
    if not lines:
        return []

    if not line_scores:
        line_end = min(len(lines), DEFAULT_SNIPPET_LINE_LIMIT)
        return [
            ContextSnippet(
                relative_path=relative_path,
                content="\n".join(lines[:line_end]).strip(),
                score=path_score,
                kind="path",
                line_start=1,
                line_end=line_end,
            )
        ]

    snippets: list[ContextSnippet] = []
    selected_ranges: list[tuple[int, int]] = []
    for line_number, score in sorted(line_scores.items(), key=lambda item: item[1], reverse=True):
        line_start, line_end = build_window(
            line_number=line_number,
            total_lines=len(lines),
            window_size=DEFAULT_SNIPPET_LINE_LIMIT,
        )
        if overlaps_existing_range(line_start, line_end, selected_ranges):
            continue

        selected_ranges.append((line_start, line_end))
        window_score = path_score + sum(
            line_score
            for current_line_number, line_score in line_scores.items()
            if line_start <= current_line_number <= line_end
        )
        snippets.append(
            ContextSnippet(
                relative_path=relative_path,
                content="\n".join(lines[line_start - 1 : line_end]).strip(),
                score=window_score,
                kind="lexical",
                line_start=line_start,
                line_end=line_end,
            )
        )
        if len(snippets) >= MAX_FILE_SNIPPETS:
            break

    return snippets


def build_window(
    *,
    line_number: int,
    total_lines: int,
    window_size: int,
) -> tuple[int, int]:
    lines_before = (window_size - 1) // 2
    line_start = max(line_number - lines_before, 1)
    line_end = min(line_start + window_size - 1, total_lines)
    line_start = max(line_end - window_size + 1, 1)
    return line_start, line_end


def overlaps_existing_range(
    line_start: int,
    line_end: int,
    existing_ranges: list[tuple[int, int]],
) -> bool:
    for existing_line_start, existing_line_end in existing_ranges:
        if line_start <= existing_line_end and line_end >= existing_line_start:
            return True

    return False


def deduplicate_snippets(snippets: list[ContextSnippet]) -> list[ContextSnippet]:
    deduped_snippets: list[ContextSnippet] = []
    seen_keys: set[tuple[str, int | None, int | None, str]] = set()
    seen_fingerprints: set[tuple[str, str]] = set()

    for snippet in snippets:
        snippet_key = (
            snippet.relative_path,
            snippet.line_start,
            snippet.line_end,
            snippet.kind,
        )
        snippet_fingerprint = (
            snippet.relative_path,
            " ".join(snippet.content.split()),
        )
        if snippet_key in seen_keys or snippet_fingerprint in seen_fingerprints:
            continue

        seen_keys.add(snippet_key)
        seen_fingerprints.add(snippet_fingerprint)
        deduped_snippets.append(snippet)

    return deduped_snippets
