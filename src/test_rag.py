from rag import similarity_search
from utils.repos import get_repo_path

REPO_NAME = 'celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0'

def main() -> None:
    results = similarity_search(
        query='function definition',
        repo_path=get_repo_path(REPO_NAME)
    )
    if not results:
        print("No similar documents found.")
        return

    top_result = results[0]
    print(f"Top match: {top_result.metadata['relative_path']}")
    print()
    print(top_result.page_content)


if __name__ == "__main__":
    main()
