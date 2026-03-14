import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_REGEX_CHUNK_LINE_LIMIT = 12


def get_regex_chunk_line_limit() -> int:
    value = os.getenv(
        "REGEX_CHUNK_LINE_LIMIT",
        str(DEFAULT_REGEX_CHUNK_LINE_LIMIT),
    )

    try:
        chunk_line_limit = int(value)
    except ValueError as error:
        raise RuntimeError("REGEX_CHUNK_LINE_LIMIT must be an integer.") from error

    if chunk_line_limit <= 0:
        raise RuntimeError("REGEX_CHUNK_LINE_LIMIT must be greater than 0.")

    return chunk_line_limit
