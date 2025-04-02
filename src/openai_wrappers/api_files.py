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


def files_list(client: OpenAI, limit=1_000) -> List[FileObject]:
    """
    List all files available in the OpenAI API.

    Args:
        client: OpenAI client instance
        limit: Maximum number of files to retrieve per request (default: 1,000)

    Returns:
        List of FileObject instances representing all available files.
    """
    # Get initial batch of files
    files = list(client.files.list(limit=limit))

    # If we hit the limit, there might be more files to fetch
    if len(files) == limit:
        while True:
            try:
                # Fetch next batch using the ID of the last file as cursor
                batch = list(client.files.list(limit=limit, after=(files[-1] or {}).get("id")))
                files.extend(batch)

                # If we got fewer files than the limit, we've reached the end
                if len(batch) != limit:
                    break
            except Exception as e:
                warn(e)
                break
    return files


def file_upload(client: OpenAI, file: FileUpload) -> FileObject:
    """
    Upload a file to the OpenAI API.

    Args:
        client: OpenAI client instance
        file: FileUpload object containing the file data and metadata

    Returns:
        FileObject instance representing the uploaded file.
    """
    # Create the file in the OpenAI API
    resp = client.files.create(file=(file.filename, file.file_data), purpose=file.purpose)
    return resp


def file_delete(client: OpenAI, file_id: str):
    """
    Delete a file from OpenAI.

    Args:
        client: The OpenAI client instance
        file_id: The ID of the file to delete

    Returns:
        The deletion status response
    """
    response = client.files.delete(file_id=file_id)
    return response