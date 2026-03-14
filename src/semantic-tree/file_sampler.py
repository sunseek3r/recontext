
import glob
import random

def get_random_files(num_files, extension, project_root="."):
    """
    Get the contents of N random files with a given extension from the repository.

    :param num_files: The number of random files to select.
    :param extension: The file extension to search for (e.g., ".py").
    :param project_root: The root directory of the project to search in.
    :return: A list of strings, where each string is the content of a randomly selected file.
    """
    if not extension.startswith("."):
        extension = "." + extension

    # Find all files with the given extension
    search_pattern = f"{project_root}/**/*{extension}"
    all_files = glob.glob(search_pattern, recursive=True)

    # Select a random sample of files
    if len(all_files) < num_files:
        # If there are fewer files than requested, return all of them
        random_sample = all_files
    else:
        random_sample = random.sample(all_files, num_files)

    # Read the content of the selected files
    file_contents = []
    for file_path in random_sample:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_contents.append(f.read())
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return file_contents


