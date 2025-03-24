from typing import Literal, List, Optional

from openai import OpenAI
from openai.types.file_object import FileObject
from dataclasses import dataclass

from core.logger import warn


@dataclass
class FileUpload:
    file_data: bytes
    filename: str
    purpose: Literal["assistants", "fine-tune", "vision", "batch"]


def files_list(limit=1_000, client: Optional[OpenAI] = None) -> List[FileObject]:
    client = client or OpenAI()

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


def file_upload(file: FileUpload, client: Optional[OpenAI] = None) -> FileObject:
    client = client or OpenAI()
    resp = client.files.create(file=(file.filename, file.file_data), purpose=file.purpose)
    return resp
