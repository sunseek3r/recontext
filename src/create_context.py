from typing import List
import src.prompts.rag_queries as rq

from src.llm.client import complete_text
from src.rag.vector_store import similarity_search
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

    get_query_functions = [
        { 
            'f': rq.get_rag_query_class_example,
            'samples_cnt': 1,
            'id': 'class-example',
        },
        {
            'f': rq.get_rag_query_function_example,
            'samples_cnt': 1,
            'id': 'function-example',
        },
        {
            'f': rq.get_rag_query_naming_convention_example,
            'samples_cnt': 1,
            'id': 'naming-convention-example',
        },
        {
            'f': rq.get_rag_query_comment_example,
            'samples_cnt': 1,
            'id': 'comment-example',
        },
        {
            'f': rq.get_rag_query_env_var_access_example,
            'samples_cnt': 1,
            'id': 'env-var-access-example',
        }
    ]

    rag_samples = []
    for entry in get_query_functions:
        samples = similarity_search(
            query=entry['f'](), 
            repo_path=repo_path,
            limit=entry['samples_cnt']
        )
        for sample in samples:
            print('[' + entry['id'] + ']')
            print(sample.page_content)
            rag_samples.append(sample.page_content)
        print('=' * 50)

    print('\n\n\n\n\n\n')


    # ===== 2. GET N RANDOM FILES =====
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


    # ===== 3. GET SUMMARY FOR RAG SAMPLES AND RANDOM FILES =====
    code_samples_summary = complete_text(f"""
Extract key code styles, naming conventions, and other important findings from this code.
DO NOT describe the logic of it.
You must be REALLY specific on describing code style in the project, with providing examples.
{"\n".join(random_files_contents + rag_samples)}
""")
    print(code_samples_summary)
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)

    # ===== 4. ANALYZE PREFIX AND SUFFIX =====

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

    # ===== 5. COMPOSE FINAL COMPLETION PROMPT =====
    final_context = compose_fim_completion_prompt(
        language=cfg.LANGUAGE,
        codebase_summary=code_samples_summary,
        prefix_suffix_summary=prefix_suffix_summary
    )
    print(final_context)
    return final_context


def compose_fim_completion_prompt(
    language: str,
    codebase_summary: str,
    prefix_suffix_summary: str
) -> str:
    return f"""
You are a software engineer with a great experience in {language}.
You must ALWAYS follow patterns established within the codebase you have to work with.
If local code rules and patterns contradicts conventional code styles, you still must obey and preserve those patterns.

Here are some findings about the codebase that you have already found after reviewing several files:
{codebase_summary}

Here is a summary of a file that you are working with:
{prefix_suffix_summary}

Your task is to create fill-in-the-middle completion for the given prefix and suffix of the file.
Your completion must complete the logic of missed piece of code and follow patterns, principles and rules established in the codebase.
"""
