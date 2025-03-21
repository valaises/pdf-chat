from typing import Literal, Iterable

from openai import OpenAI
from dataclasses import dataclass

from core.logger import warn


@dataclass
class FileUpload:
    file_data: bytes
    filename: str
    purpose: Literal["assistants", "fine-tune", "vision", "batch"]


async def files_list(limit=1_000):
    client = OpenAI()

    files = list(client.files.list(limit=limit))
    if len(files) == limit:
        while True:
            try:
                batch = list(client.files.list(limit=limit, after=(files[-1] or {}).get("id")))
                files.extend(batch)
                if len(batch) != limit:
                    break
            except Exception as e:
                warn(e)
                break
    return files


async def files_upload(files: Iterable[FileUpload]):
    client = OpenAI()
    responses = []
    for file in files:
        resp = client.files.create(file=(file.filename, file.file_data), purpose=file.purpose)
        responses.append(resp)
    return responses
