import os 
import src.config as cfg

def get_repo_path(repo_name: str) -> str:
    """
    Returns a path to a given repository by its name, using current language and stage set in config

    :param repo_name: for example "0d3b1e254f9178828f62b7b84f0307882e28e2a0"
    """
    return os.path.join(
        cfg.REPOS_DIR_PATH,
        repo_name
    )


