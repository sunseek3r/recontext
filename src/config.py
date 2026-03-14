import os 


# ========= TO CHANGE =========

LANGUAGE = 'python' # 'python' | 'kotlin'
STAGE = 'practice'
CUR_REPO_NAME = 'celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0'

# ========= TO CHANGE =========

DATA_DIR_PATH = 'data'

REPOS_DIR_PATH = os.path.join(DATA_DIR_PATH, f'repositories-{LANGUAGE}-{STAGE}')

REPO_PATH = os.path.join(REPOS_DIR_PATH, CUR_REPO_NAME)
