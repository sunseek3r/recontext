from utils.repos import get_repo_path
import prompts.rag_queries as rq
from rag.vector_store import similarity_search

def create_context_for_repo(
    repo_name: str,
    file_prefix: str,
    file_suffix: str,
): 
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
