from rag import similarity_search
from utils.repos import get_repo_path

# REPO_NAME = 'celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0'
REPO_NAME = 'pallets__werkzeug-7fd02ca598a29681f1c0b27377b9751f9c8f8ce0'

def main() -> None:
    results = similarity_search(
        query='class',
        repo_path=get_repo_path(REPO_NAME),
        limit=5
    )
    if not results:
        print("No similar documents found.")
        return

    for result in results:
        print('\n' + '=' * 50 + '\n')
        print(result.page_content)
        print('\n' + '=' * 50 + '\n')


if __name__ == "__main__":
    main()
