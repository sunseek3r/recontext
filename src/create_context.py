from typing import List
import src.prompts.rag_queries as ragq
import src.prompts.regex_queries as rq

from src.llm.client import complete_text
from src.rag.vector_store import similarity_search
from src.regex_search import regex_search
from src.utils.file_sampler import get_random_files
from src.utils.repos import get_repo_path
import src.config as cfg


def create_context_for_repo(
    repo_name: str,
    file_prefix: str,
    file_suffix: str,
):
    # ===== 1. EXTRACTING CODE EXAMPLES VIA RAG =====
    repo_path = get_repo_path(repo_name)

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

    rag_samples = []
    seen_samples = set()
    for entry in get_rag_query_functions:
        samples = similarity_search(
            query=entry['f'](),
            repo_path=repo_path,
            limit=entry['samples_cnt']
        )

        for sample in samples:
            if sample.page_content in seen_samples:
                continue

            seen_samples.add(sample.page_content)
            print('[' + entry['id'] + ']')
            print(sample.page_content)
            rag_samples.append(sample.page_content)
        print('=' * 50)

    print('=' * 50)
    print('=' * 50)
    print('=' * 50)
    print('\n\n\n\n\n\n')


    # ===== 2. EXTRACTING CODE EXAMPLES VIA REGEX SEARCH =====
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

    regex_samples = []
    for entry in get_regex_functions:
        samples = regex_search(
            pattern=entry['f'](),
            repo_path=repo_path,
            limit=entry['samples_cnt']
        )
        for sample in samples:
            if sample.page_content in seen_samples:
                continue

            seen_samples.add(sample.page_content)
            print('[' + entry['id'] + ']')
            print(sample.page_content)
            regex_samples.append(sample.page_content)
        print('=' * 50)

    print('=' * 50)
    print('=' * 50)
    print('=' * 50)
    print('\n\n\n\n\n\n')


    # ===== 3. GET N RANDOM FILES =====
    random_files_contents = get_random_files(
        num_files=3,
        extension='.py',
        project_root=repo_path
    )
    for file in random_files_contents:
        print(file)
        print('\n')
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)


    # ===== 4. GET SUMMARY FOR SAMPLES AND RANDOM FILES =====
    code_samples_summary = complete_text(f"""
Extract key code styles, naming conventions, and other important findings from this code.
DO NOT describe the logic of it.
You must be REALLY specific on describing code style in the project, with providing examples.
{"\n".join(random_files_contents + rag_samples + regex_samples)}
""")
    print(code_samples_summary)
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)

    # ===== 5. ANALYZE PREFIX AND SUFFIX =====

    prefix_suffix_summary = complete_text(f"""
You are given the beginning and the end of the file, with some relatively small part missing. You have to figure out and explain what is going in this file in details.
```
{file_prefix}
...
{file_suffix}
```
""")
    print(prefix_suffix_summary)
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)

    # ===== 6. COMPOSE FINAL COMPLETION PROMPT =====
    final_context = compose_fim_completion_prompt(
        language=cfg.LANGUAGE,
        codebase_summary=code_samples_summary,
        prefix_suffix_summary=prefix_suffix_summary,
        examples=regex_samples
    )
    print(final_context)
    return final_context


def compose_fim_completion_prompt(
    language: str,
    codebase_summary: str | None,
    prefix_suffix_summary: str | None,
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

    task_description_part = f"""
Your task is to create fill-in-the-middle completion for the given prefix and suffix of the file.
Your completion must complete the logic of missed piece of code and follow patterns, principles and rules established in the codebase.
"""


    return f"""
{intro_part}

{'' if codebase_summary is None else summary_part}

{'' if prefix_suffix_summary is None else prefix_suffix_summary_part }

{'' if examples is None else examples_part}

{task_description_part}
"""
