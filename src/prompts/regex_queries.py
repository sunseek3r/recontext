import src.config as cfg


def get_regex_query_class_example():
    if cfg.LANGUAGE == "kotlin":
        return r"^\s*(?:data\s+|sealed\s+|enum\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*"

    return r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*"


def get_regex_query_function_example():
    if cfg.LANGUAGE == "kotlin":
        return r"^\s*fun\s+[A-Za-z_][A-Za-z0-9_]*\s*\("

    return r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\("


def get_regex_query_naming_convention_example():
    if cfg.LANGUAGE == "kotlin":
        return r"\b(?:val|var)\s+[a-z][A-Za-z0-9]*\s*(?::[^=]+)?="

    return r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\s*="


def get_regex_query_comment_example():
    if cfg.LANGUAGE == "kotlin":
        return r"^\s*//\s+\S+"

    return r"^\s*#\s+\S+"


def get_regex_query_env_var_access_example():
    if cfg.LANGUAGE == "kotlin":
        return r"\bSystem\.getenv\s*\("

    return r"\bos\.(?:getenv\s*\(|environ(?:\.get\s*\(|\s*\[))"
