from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any, List, TypeVar
import src.prompts.rag_queries as ragq
import src.prompts.regex_queries as rq

from src.llm.client import complete_text
from src.rag.vector_store import similarity_search
from src.regex_search import regex_search
from src.utils.context_files import compose_related_files_context, load_modified_file_examples
from src.utils.file_sampler import get_random_files
from src.utils.repos import get_repo_path
import src.config as cfg

T = TypeVar('T')


def _print_stage_break(*, enabled: bool) -> None:
    if not enabled:
        return

    print('=' * 50)
    print('=' * 50)
    print('=' * 50)
    print('\n\n\n\n\n\n')


def _print_summary(summary: str, *, enabled: bool) -> None:
    if not enabled:
        return

    print(summary)
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)


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

    print(f'[timing] {step_name}: {duration_seconds:.3f}s')


def _print_timing_summary(timings: list[tuple[str, float]], *, enabled: bool) -> None:
    if not enabled:
        return

    print('[timing] summary (slowest first):')
    for step_name, duration_seconds in sorted(
        timings,
        key=lambda timing: timing[1],
        reverse=True,
    ):
        print(f'[timing] {step_name}: {duration_seconds:.3f}s')


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
    use_rag: bool = True,
    use_regex: bool = True,
    use_random_files: bool = True,
    use_modified_files: bool = True,
    summarize_code_samples: bool = True,
    summarize_prefix_suffix: bool = True,
    verbose: bool = False,
):
    total_started_at = perf_counter()
    timings: list[tuple[str, float]] = []
    should_output = verbose

    repo_path_started_at = perf_counter()
    repo_path = get_repo_path(repo_name)
    _record_timing(
        timings,
        'resolve_repo_path',
        perf_counter() - repo_path_started_at,
        enabled=should_output,
    )
    rag_samples = []
    seen_samples = set()
    regex_samples = []
    random_files_contents = []
    modified_file_examples = []
    related_files_context = None

    if use_rag:
        rag_stage_started_at = perf_counter()
        get_rag_query_functions = [
            {
                'f': ragq.get_rag_query_class_example,
                'samples_cnt': 1,
                'id': 'class-example',
            },
            {
                'f': ragq.get_rag_query_function_example,
                'samples_cnt': 1,
                'id': 'function-example',
            },
            {
                'f': ragq.get_rag_query_naming_convention_example,
                'samples_cnt': 1,
                'id': 'naming-convention-example',
            },
            {
                'f': ragq.get_rag_query_comment_example,
                'samples_cnt': 1,
                'id': 'comment-example',
            },
            {
                'f': ragq.get_rag_query_env_var_access_example,
                'samples_cnt': 1,
                'id': 'env-var-access-example',
            }
        ]

        for entry in get_rag_query_functions:
            query_started_at = perf_counter()
            samples = similarity_search(
                query=entry['f'](),
                repo_path=repo_path,
                limit=entry['samples_cnt']
            )
            _record_timing(
                timings,
                f"rag:{entry['id']}",
                perf_counter() - query_started_at,
                enabled=should_output,
            )

            for sample in samples:
                if sample.page_content in seen_samples:
                    continue

                seen_samples.add(sample.page_content)
                if should_output:
                    print('[' + entry['id'] + ']')
                    print(sample.page_content)
                rag_samples.append(sample.page_content)
            if should_output:
                print('=' * 50)

        _record_timing(
            timings,
            'rag_total',
            perf_counter() - rag_stage_started_at,
            enabled=should_output,
        )
        _print_stage_break(enabled=should_output)

    if use_regex:
        regex_stage_started_at = perf_counter()
        get_regex_functions = [
            {
                'f': rq.get_regex_query_class_example,
                'samples_cnt': 1,
                'id': 'regex-class-example',
            },
            {
                'f': rq.get_regex_query_function_example,
                'samples_cnt': 1,
                'id': 'regex-function-example',
            },
            {
                'f': rq.get_regex_query_naming_convention_example,
                'samples_cnt': 1,
                'id': 'regex-naming-convention-example',
            },
            {
                'f': rq.get_regex_query_comment_example,
                'samples_cnt': 1,
                'id': 'regex-comment-example',
            },
            {
                'f': rq.get_regex_query_env_var_access_example,
                'samples_cnt': 1,
                'id': 'regex-env-var-access-example',
            },
        ]

        for entry in get_regex_functions:
            pattern_started_at = perf_counter()
            samples = regex_search(
                pattern=entry['f'](),
                repo_path=repo_path,
                limit=entry['samples_cnt']
            )
            _record_timing(
                timings,
                f"regex:{entry['id']}",
                perf_counter() - pattern_started_at,
                enabled=should_output,
            )
            for sample in samples:
                if sample.page_content in seen_samples:
                    continue

                seen_samples.add(sample.page_content)
                if should_output:
                    print('[' + entry['id'] + ']')
                    print(sample.page_content)
                regex_samples.append(sample.page_content)
            if should_output:
                print('=' * 50)

        _record_timing(
            timings,
            'regex_total',
            perf_counter() - regex_stage_started_at,
            enabled=should_output,
        )
        _print_stage_break(enabled=should_output)

    if use_random_files:
        random_files_started_at = perf_counter()
        random_files_contents = get_random_files(
            num_files=3,
            extension='.py',
            project_root=repo_path
        )
        _record_timing(
            timings,
            'random_files',
            perf_counter() - random_files_started_at,
            enabled=should_output,
        )
        if should_output:
            for file in random_files_contents:
                print(file)
                print('\n')
            print('=' * 50)
            print('=' * 50)
            print('=' * 50)

    if use_modified_files and modified_files:
        modified_files_started_at = perf_counter()
        modified_file_examples = load_modified_file_examples(
            repo_path,
            modified_files,
            seen_samples,
        )
        _record_timing(
            timings,
            'modified_files',
            perf_counter() - modified_files_started_at,
            enabled=should_output,
        )
        if should_output:
            for file_example in modified_file_examples:
                print('[modified]')
                print(file_example)
                print('=' * 50)
        related_files_context = compose_related_files_context(modified_file_examples) or None

    prompt_build_started_at = perf_counter()
    code_samples = random_files_contents + rag_samples + regex_samples + modified_file_examples
    code_samples_summary_prompt = None
    if summarize_code_samples and code_samples:
        code_samples_summary_prompt = f"""
Extract key code styles, naming conventions, and other important findings from this code.
DO NOT describe the logic of it.
You must be REALLY specific on describing code style in the project, with providing examples (examples are mandatory!).
{"\n".join(code_samples)}
"""

    prefix_suffix_summary_prompt = None
    if summarize_prefix_suffix:
        prefix_suffix_summary_prompt = f"""
You are given the beginning and the end of the file, with some relatively small part missing. You have to figure out and explain what is going in this file in details.
```
{file_prefix}
...
{file_suffix}
```
"""
    _record_timing(
        timings,
        'build_summary_prompts',
        perf_counter() - prompt_build_started_at,
        enabled=should_output,
    )

    code_samples_summary = None
    prefix_suffix_summary = None

    llm_stage_started_at = perf_counter()
    futures: dict[str, Future[tuple[str, str, float]]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if code_samples_summary_prompt is not None:
            futures['code_samples_summary'] = executor.submit(
                _run_timed_step,
                'llm:code_samples_summary',
                complete_text,
                code_samples_summary_prompt,
            )

        if prefix_suffix_summary_prompt is not None:
            futures['prefix_suffix_summary'] = executor.submit(
                _run_timed_step,
                'llm:prefix_suffix_summary',
                complete_text,
                prefix_suffix_summary_prompt,
            )

        if 'code_samples_summary' in futures:
            code_samples_summary, step_name, duration_seconds = futures['code_samples_summary'].result()
            _record_timing(
                timings,
                step_name,
                duration_seconds,
                enabled=should_output,
            )
            _print_summary(code_samples_summary, enabled=should_output)

        if 'prefix_suffix_summary' in futures:
            prefix_suffix_summary, step_name, duration_seconds = futures['prefix_suffix_summary'].result()
            _record_timing(
                timings,
                step_name,
                duration_seconds,
                enabled=should_output,
            )
            _print_summary(prefix_suffix_summary, enabled=should_output)
    _record_timing(
        timings,
        'llm_total_wall_time',
        perf_counter() - llm_stage_started_at,
        enabled=should_output,
    )

    final_context_started_at = perf_counter()
    final_context = compose_fim_completion_prompt(
        language=cfg.LANGUAGE,
        codebase_summary=code_samples_summary,
        prefix_suffix_summary=prefix_suffix_summary,
        related_files=related_files_context,
        examples=regex_samples or None,
    )
    _record_timing(
        timings,
        'compose_final_context',
        perf_counter() - final_context_started_at,
        enabled=should_output,
    )
    if should_output:
        print(final_context)
    _record_timing(
        timings,
        'create_context_for_repo_total',
        perf_counter() - total_started_at,
        enabled=should_output,
    )
    _print_timing_summary(timings, enabled=should_output)
    return final_context


def compose_fim_completion_prompt(
    language: str,
    codebase_summary: str | None,
    prefix_suffix_summary: str | None,
    related_files: str | None,
    examples: List[str] | None,
) -> str:
    intro_part = f"""
You are a software engineer with a great experience in {language}.
You must ALWAYS follow patterns established within the codebase you have to work with.
If local code rules and patterns contradicts conventional code styles, you still must obey and preserve those patterns.
"""

    summary_part = f"""
Here are some findings about the codebase that you have already found after reviewing several files:
{codebase_summary}
"""

    prefix_suffix_summary_part = f"""
Here is a summary of a file that you are working with:
{prefix_suffix_summary}
"""

    examples_part = f"""
Here are some examples of code from the codebase that may help you:
{'\n\n'.join([] if examples is None else examples)}
"""

    related_files_part = f"""
Here are code that may be potentially related to the file you are working with:
{related_files}
"""

    task_description_part = f"""
=== TASK ===

Your task is to create fill-in-the-middle completion for the given prefix and suffix of the file.
Your completion must complete the logic of missed piece of code and follow patterns, principles and rules established in the codebase.
"""


    return f"""
{intro_part}

{'' if codebase_summary is None else summary_part}

{'' if prefix_suffix_summary is None else prefix_suffix_summary_part }

{'' if examples is None else examples_part}

{'' if related_files is None else related_files_part}

{task_description_part}
"""
