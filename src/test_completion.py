from llm import complete_text



def main() -> None:
  result = complete_text("Write a short hello-world function in Python.")
  print(result)


if __name__ == "__main__":
    main()
