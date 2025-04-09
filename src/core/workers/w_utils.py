import hashlib
from typing import Iterator, Dict
from pathlib import Path

import ujson as json


def jsonl_reader(file: Path) -> Iterator[Dict]:
    with file.open("r") as f:
        for line in f:
            yield json.loads(line)


def generate_paragraph_id(paragraph_text: str) -> str:
    """
    Generate a unique paragraph ID based on the paragraph text.

    Args:
        paragraph_text: The text of the paragraph

    Returns:
        A unique paragraph ID with format pid-{hash}
    """
    return f"pid-{generate_content_hash(paragraph_text, length=8)}"


def generate_content_hash(content: str, salt: str = "", length: int = 16) -> str:
    """
    Generate a hash from content with optional salt.

    Args:
        content: The content to hash
        salt: Optional salt to add to the content before hashing
        length: Length of the hash to return

    Returns:
        A hexadecimal hash string of the specified length
    """
    # noinspection PyTypeChecker
    return hashlib.md5((salt + content).encode()).hexdigest()[:length]
