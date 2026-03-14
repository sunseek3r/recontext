from typing import List
from llm.client import complete_text
from utils.repos import get_repo_path
from utils.file_sampler import get_random_files
import prompts.rag_queries as rq
from rag.vector_store import similarity_search

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
        },
        { 
            'f': rq.get_rag_query_class_example,
            'samples_cnt': 1,
        },
        {
            'f': rq.get_rag_query_function_example,
            'samples_cnt': 1,
        },
        {
            'f': rq.get_rag_query_naming_convention_example,
            'samples_cnt': 1,
        },
        {
            'f': rq.get_rag_query_comment_example,
            'samples_cnt': 1,
        },
        {
            'f': rq.get_rag_query_env_var_access_example,
            'samples_cnt': 1,
        }
    ]

    for entry in get_query_functions:
        samples = similarity_search(
            query=entry['f'](), 
            repo_path=repo_path,
            limit=entry['samples_cnt']
        )
        print(samples)
        print('=' * 50)

    print('\n\n\n\n\n\n')


    # ===== 2. GET N RANDOM FILES AND SUMMARIZE CODE STYLE WITH LLM =====
    files_contents = get_random_files(
        num_files=5,
        extension='.py',
        project_root=repo_path
    )
    for file in files_contents:
        print(file)
        print('\n')
    print('=' * 50)
    print('=' * 50)
    print('=' * 50)

    random_files_summary = complete_text(f"""
Extract key code styles, naming conventions, and other importantn findings from this code.
DO NOT describe the logic of it.
{"\n".join(files_contents)}
""")
    print(random_files_summary)

    return "\n".join(files_contents)

    # ===== 3. ANALYZE PREFIX AND SUFFIX =====

    # ===== COMPOSE FINAL COMPLETION PROMPT =====
    return compose_fim_completion_prompt(
        language=cfg.
    )


def compose_fim_completion_prompt(
    language: str,
    examples: List[str],
    codebase_summary: str,
) -> str:
    return f"""
You are a software engineer with a great experience in {language}.
You must ALWAYS follow patterns established with the codebase you have to work with.
If local code rules and patterns contradicts conventional code styles, you still must obey and preserve those patterns.

Here are some examples that may be useful for you to understand the codebase you are working in:
{examples}

Here are some findings about the codebase that you have already found after reviewing several files:
{codebase_summary}
"""
