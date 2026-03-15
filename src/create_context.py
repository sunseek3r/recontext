from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any, TypeVar

import src.prompts.rag_queries as ragq
import src.prompts.regex_queries as rq
from src.llm.client import complete_text
from src.rag.vector_store import similarity_search
from src.regex_search import regex_search
from src.utils.context_files import load_modified_file_examples
from src.utils.context_snippets import (
    ContextSnippet,
    collect_targeted_snippets,
    compose_context_snippets,
    document_to_snippet,
)
from src.utils.file_sampler import get_random_files
from src.utils.repos import get_repo_path

T = TypeVar("T")

SUMMARY_SEPARATOR = "\n\n"


def _print_stage_break(*, enabled: bool) -> None:
    if not enabled:
        return

    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    print("\n\n\n\n\n\n")


def _print_summary(summary: str, *, enabled: bool) -> None:
    if not enabled:
        return

    print(summary)
    print("=" * 50)
    print("=" * 50)
    print("=" * 50)


def _record_timing(
    timings: list[tuple[str, float]],
    step_name: str,
    duration_seconds: float,
    *,
    enabled: bool,
) -> None:
    timings.append((step_name, duration_seconds))
    if not enabled:
        return

    print(f"[timing] {step_name}: {duration_seconds:.3f}s")


def _print_timing_summary(timings: list[tuple[str, float]], *, enabled: bool) -> None:
    if not enabled:
        return

    print("[timing] summary (slowest first):")
    for step_name, duration_seconds in sorted(
        timings,
        key=lambda timing: timing[1],
        reverse=True,
    ):
        print(f"[timing] {step_name}: {duration_seconds:.3f}s")


def _run_timed_step(step_name: str, fn: Any, *args: Any) -> tuple[T, str, float]:
    started_at = perf_counter()
    result = fn(*args)
    return result, step_name, perf_counter() - started_at


def create_context_for_repo(
    repo_name: str,
    file_prefix: str,
    file_suffix: str,
    modified_files: list[str] | None = None,
    *,
    target_file_path: str | None = None,
    use_rag: bool = True,
    use_regex: bool = True,
    use_random_files: bool = True,
    use_modified_files: bool = True,
    summarize_code_samples: bool = True,
    summarize_prefix_suffix: bool = True,
    max_context_chars: int | None = None,
    verbose: bool = False,
) -> str:
    total_started_at = perf_counter()
    timings: list[tuple[str, float]] = []
    should_output = verbose

    repo_path_started_at = perf_counter()
    repo_path = get_repo_path(repo_name)
    _record_timing(
        timings,
        "resolve_repo_path",
        perf_counter() - repo_path_started_at,
        enabled=should_output,
    )

    seen_samples: set[str] = set()
    modified_file_snippets: list[ContextSnippet] = []
    auxiliary_snippets: list[ContextSnippet] = []
    summary_sources: list[str] = []

    if use_modified_files and modified_files:
        modified_files_started_at = perf_counter()
        modified_file_snippets = load_modified_file_examples(
            repo_path,
            modified_files,
            seen_samples,
        )
        _record_timing(
            timings,
            "modified_files",
            perf_counter() - modified_files_started_at,
            enabled=should_output,
        )
        if should_output:
            for snippet in modified_file_snippets:
                print("[modified]")
                print(snippet.render())
                print("=" * 50)

    excluded_paths = {
        snippet.relative_path
        for snippet in modified_file_snippets
    }
    if target_file_path is not None:
        excluded_paths.add(target_file_path)

    if use_regex:
        targeted_retrieval_started_at = perf_counter()
        targeted_snippets = collect_targeted_snippets(
            repo_path,
            file_prefix=file_prefix,
            file_suffix=file_suffix,
            target_file_path=target_file_path,
            exclude_relative_paths=excluded_paths,
            limit=8,
        )
        for snippet in targeted_snippets:
            seen_samples.add(snippet.content)

        auxiliary_snippets.extend(targeted_snippets)
        summary_sources.extend(snippet.render() for snippet in targeted_snippets)
        _record_timing(
            timings,
            "targeted_retrieval",
            perf_counter() - targeted_retrieval_started_at,
            enabled=should_output,
        )
        if should_output:
            for snippet in targeted_snippets:
                print("[targeted]")
                print(snippet.render())
                print("=" * 50)

        regex_stage_started_at = perf_counter()
        regex_documents = collect_regex_documents(repo_path=repo_path, enabled=use_regex)
        regex_snippets = [
            document_to_snippet(document, score=1, kind="regex")
            for document in regex_documents
            if document.page_content not in seen_samples
        ]
        for snippet in regex_snippets:
            seen_samples.add(snippet.content)

        auxiliary_snippets.extend(regex_snippets)
        summary_sources.extend(snippet.render() for snippet in regex_snippets)
        _record_timing(
            timings,
            "regex_total",
            perf_counter() - regex_stage_started_at,
            enabled=should_output,
        )
        if should_output:
            for snippet in regex_snippets:
                print("[regex]")
                print(snippet.render())
                print("=" * 50)
        _print_stage_break(enabled=should_output)

    if use_rag:
        rag_stage_started_at = perf_counter()
        rag_documents = collect_rag_documents(repo_path=repo_path)
        rag_snippets = [
            document_to_snippet(document, score=2, kind="rag")
            for document in rag_documents
            if document.page_content not in seen_samples
        ]
        for snippet in rag_snippets:
            seen_samples.add(snippet.content)

        auxiliary_snippets.extend(rag_snippets)
        summary_sources.extend(snippet.render() for snippet in rag_snippets)
        _record_timing(
            timings,
            "rag_total",
            perf_counter() - rag_stage_started_at,
            enabled=should_output,
        )
        if should_output:
            for snippet in rag_snippets:
                print("[rag]")
                print(snippet.render())
                print("=" * 50)
        _print_stage_break(enabled=should_output)

    if use_random_files:
        random_files_started_at = perf_counter()
        random_files_contents = get_random_files(
            num_files=3,
            extension=".py",
            project_root=repo_path,
        )
        summary_sources.extend(random_files_contents)
        _record_timing(
            timings,
            "random_files",
            perf_counter() - random_files_started_at,
            enabled=should_output,
        )
        if should_output:
            for file_content in random_files_contents:
                print("[random]")
                print(file_content)
                print("=" * 50)

    prompt_build_started_at = perf_counter()
    code_samples_summary_prompt = None
    if summarize_code_samples and summary_sources:
        code_samples_summary_prompt = build_code_samples_summary_prompt(summary_sources)

    prefix_suffix_summary_prompt = None
    if summarize_prefix_suffix:
        prefix_suffix_summary_prompt = build_prefix_suffix_summary_prompt(
            file_prefix=file_prefix,
            file_suffix=file_suffix,
        )
    _record_timing(
        timings,
        "build_summary_prompts",
        perf_counter() - prompt_build_started_at,
        enabled=should_output,
    )

    code_samples_summary = None
    prefix_suffix_summary = None

    llm_stage_started_at = perf_counter()
    futures: dict[str, Future[tuple[str, str, float]]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if code_samples_summary_prompt is not None:
            futures["code_samples_summary"] = executor.submit(
                _run_timed_step,
                "llm:code_samples_summary",
                complete_text,
                code_samples_summary_prompt,
            )

        if prefix_suffix_summary_prompt is not None:
            futures["prefix_suffix_summary"] = executor.submit(
                _run_timed_step,
                "llm:prefix_suffix_summary",
                complete_text,
                prefix_suffix_summary_prompt,
            )

        if "code_samples_summary" in futures:
            code_samples_summary, step_name, duration_seconds = futures["code_samples_summary"].result()
            _record_timing(
                timings,
                step_name,
                duration_seconds,
                enabled=should_output,
            )
            _print_summary(code_samples_summary, enabled=should_output)

        if "prefix_suffix_summary" in futures:
            prefix_suffix_summary, step_name, duration_seconds = futures["prefix_suffix_summary"].result()
            _record_timing(
                timings,
                step_name,
                duration_seconds,
                enabled=should_output,
            )
            _print_summary(prefix_suffix_summary, enabled=should_output)
    _record_timing(
        timings,
        "llm_total_wall_time",
        perf_counter() - llm_stage_started_at,
        enabled=should_output,
    )

    final_context_started_at = perf_counter()
    final_context = compose_fim_completion_prompt(
        codebase_summary=code_samples_summary,
        prefix_suffix_summary=prefix_suffix_summary,
        auxiliary_snippets=auxiliary_snippets,
        modified_file_snippets=modified_file_snippets,
        max_context_chars=max_context_chars,
    )
    _record_timing(
        timings,
        "compose_final_context",
        perf_counter() - final_context_started_at,
        enabled=should_output,
    )
    if should_output:
        print(final_context)
    _record_timing(
        timings,
        "create_context_for_repo_total",
        perf_counter() - total_started_at,
        enabled=should_output,
    )
    _print_timing_summary(timings, enabled=should_output)
    return final_context


def collect_rag_documents(repo_path: str) -> list[Any]:
    rag_documents: list[Any] = []
    seen_contents: set[str] = set()
    get_rag_query_functions = [
        {
            "f": ragq.get_rag_query_class_example,
            "samples_cnt": 1,
        },
        {
            "f": ragq.get_rag_query_function_example,
            "samples_cnt": 1,
        },
        {
            "f": ragq.get_rag_query_naming_convention_example,
            "samples_cnt": 1,
        },
        {
            "f": ragq.get_rag_query_comment_example,
            "samples_cnt": 1,
        },
        {
            "f": ragq.get_rag_query_env_var_access_example,
            "samples_cnt": 1,
        },
    ]

    for entry in get_rag_query_functions:
        samples = similarity_search(
            query=entry["f"](),
            repo_path=repo_path,
            limit=entry["samples_cnt"],
        )
        for sample in samples:
            if sample.page_content in seen_contents:
                continue

            seen_contents.add(sample.page_content)
            rag_documents.append(sample)

    return rag_documents


def collect_regex_documents(repo_path: str, *, enabled: bool) -> list[Any]:
    if not enabled:
        return []

    regex_documents: list[Any] = []
    seen_documents: set[tuple[str, int | None, int | None]] = set()
    get_regex_functions = [
        rq.get_regex_query_class_example,
        rq.get_regex_query_function_example,
        rq.get_regex_query_naming_convention_example,
        rq.get_regex_query_comment_example,
        rq.get_regex_query_env_var_access_example,
    ]

    for get_regex_query in get_regex_functions:
        samples = regex_search(
            pattern=get_regex_query(),
            repo_path=repo_path,
            limit=1,
        )
        for sample in samples:
            sample_key = (
                sample.metadata["relative_path"],
                sample.metadata.get("line_start"),
                sample.metadata.get("line_end"),
            )
            if sample_key in seen_documents:
                continue

            seen_documents.add(sample_key)
            regex_documents.append(sample)

    return regex_documents


def build_code_samples_summary_prompt(summary_sources: list[str]) -> str:
    rendered_sources = "\n".join(summary_sources)
    return f"""
Extract compact coding patterns that are likely useful for fill-in-the-middle completion.
Focus only on naming, imports, API usage, assertions, and error-handling patterns.
Keep the answer short and concrete.
{rendered_sources}
"""


def build_prefix_suffix_summary_prompt(
    *,
    file_prefix: str,
    file_suffix: str,
) -> str:
    return f"""
You are given the beginning and the end of a file with a missing middle.
List only the concrete constraints for the missing span:
- active class or function
- referenced symbols
- expected shape of the missing code
- indentation and scope constraints
Keep the answer short.
```
{file_prefix}
...
{file_suffix}
```
"""


def compose_fim_completion_prompt(
    *,
    codebase_summary: str | None,
    prefix_suffix_summary: str | None,
    auxiliary_snippets: list[ContextSnippet],
    modified_file_snippets: list[ContextSnippet],
    max_context_chars: int | None,
) -> str:
    context_blocks: list[str] = []

    summary_block = compose_summary_block(
        codebase_summary=codebase_summary,
        prefix_suffix_summary=prefix_suffix_summary,
    )
    if summary_block:
        context_blocks.append(summary_block)

    ranked_auxiliary_snippets = sorted(
        deduplicate_context_snippets(auxiliary_snippets),
        key=lambda snippet: (snippet.score, snippet.relative_path, snippet.line_start or 0),
    )
    if ranked_auxiliary_snippets:
        context_blocks.append(compose_context_snippets(ranked_auxiliary_snippets))

    if modified_file_snippets:
        context_blocks.append(compose_context_snippets(modified_file_snippets))

    return pack_context_blocks(context_blocks, max_context_chars=max_context_chars)


def compose_summary_block(
    *,
    codebase_summary: str | None,
    prefix_suffix_summary: str | None,
) -> str:
    summary_parts: list[str] = []
    if codebase_summary:
        summary_parts.append(codebase_summary.strip())

    if prefix_suffix_summary:
        summary_parts.append(prefix_suffix_summary.strip())

    if not summary_parts:
        return ""

    return SUMMARY_SEPARATOR.join(summary_parts).strip()


def deduplicate_context_snippets(snippets: list[ContextSnippet]) -> list[ContextSnippet]:
    deduplicated_snippets: list[ContextSnippet] = []
    seen_keys: set[tuple[str, int | None, int | None, str]] = set()
    seen_contents: set[tuple[str, str]] = set()

    for snippet in snippets:
        snippet_key = (
            snippet.relative_path,
            snippet.line_start,
            snippet.line_end,
            snippet.kind,
        )
        snippet_content_key = (
            snippet.relative_path,
            " ".join(snippet.content.split()),
        )
        if snippet_key in seen_keys or snippet_content_key in seen_contents:
            continue

        seen_keys.add(snippet_key)
        seen_contents.add(snippet_content_key)
        deduplicated_snippets.append(snippet)

    return deduplicated_snippets


def pack_context_blocks(
    context_blocks: list[str],
    *,
    max_context_chars: int | None,
) -> str:
    normalized_blocks = [block for block in context_blocks if block]
    if max_context_chars is None:
        return "".join(normalized_blocks)

    selected_blocks: list[str] = []
    total_chars = 0
    for block in reversed(normalized_blocks):
        if total_chars + len(block) <= max_context_chars:
            selected_blocks.append(block)
            total_chars += len(block)
            continue

        if not selected_blocks:
            selected_blocks.append(truncate_block_from_left(block, max_context_chars))
        break

    return "".join(reversed(selected_blocks))


def truncate_block_from_left(block: str, max_context_chars: int) -> str:
    if len(block) <= max_context_chars:
        return block

    first_newline_index = block.find("\n")
    if first_newline_index == -1 or first_newline_index >= max_context_chars:
        return block[-max_context_chars:]

    header = block[: first_newline_index + 1]
    remaining_chars = max_context_chars - len(header)
    if remaining_chars <= 0:
        return block[-max_context_chars:]

    return header + block[-remaining_chars:]
