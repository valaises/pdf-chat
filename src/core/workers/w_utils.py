from typing import Iterator, Dict
from pathlib import Path

import ujson as json


def jsonl_reader(file: Path) -> Iterator[Dict]:
    with file.open("r") as f:
        for line in f:
            yield json.loads(line)
