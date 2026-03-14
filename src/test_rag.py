from rag import similarity_search
from rag.settings import get_rag_query


def main() -> None:
    results = similarity_search(get_rag_query())
    if not results:
        print("No similar documents found.")
        return

    top_result = results[0]
    print(f"Top match: {top_result.metadata['relative_path']}")
    print()
    print(top_result.page_content)


if __name__ == "__main__":
    main()
