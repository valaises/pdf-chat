import os

from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import aiofiles

from fastapi import APIRouter, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.globals import FILES_DIR
from core.logger import exception
from core.repositories.repo_files import FilesRepository, FileItem


class DeleteFilePost(BaseModel):
    file_name: str


class FilesListPost(BaseModel):
    user_id: Optional[int]


class FilesRouter(APIRouter):
    def __init__(
            self,
            files_repository: FilesRepository,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._files_repository = files_repository

        self.add_api_route("/v1/file-create", self._create_file, methods=["POST"])
        self.add_api_route("/v1/file-delete", self._delete_file, methods=["POST"])
        self.add_api_route("/v1/files-list", self._list_files, methods=["POST"])
        self.add_api_route("/v1/file-upload", self._file_upload, methods=["POST"])

    async def _create_file(self, file: FileItem):
        if not FILES_DIR.joinpath(file.file_name).is_file():
            return Response(status_code=404, content={"message": "file not found"})

        resp = await self._files_repository.create_file(file)
        if resp:
            return Response(status_code=201, content={"message": "added record of file in DB"})
        else:
            return Response(status_code=400, content={"message": "failed to add record of file in DB"})

    async def _delete_file(self, post: DeleteFilePost):
        resp = await self._files_repository.delete_file(post.file_name)
        if resp:
            return Response(status_code=200, content={"message": "record of file deleted"})
        else:
            return Response(status_code=404, content={"message": "failed to delete record of file"})

    async def _list_files(self, post: FilesListPost):
        if post.user_id is not None:
            files = await self._files_repository.get_files_by_filter("user_id = ?", (post.user_id,))
        else:
            files = await self._files_repository.get_all_files()

        return Response(status_code=200, content={"files": [file.model_dump() for file in files]})

    async def _file_upload(self, request: Request):
        file_name = request.headers.get('X-File-Name')
        user_id = request.headers.get('X-User-Id')

        if user_id is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "X-User-Id header is either missing or empty"}
            )

        try:
            user_id = int(user_id)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"X-User-Id header should be convertable to int type: {str(e)}"}
            )

        if not file_name:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "X-File-Name header is either missing or empty"}
            )

        file_name = unquote(file_name)

        # Create a unique hash based on the original filename and a random UUID
        random_hash = hashlib.sha256(f"{file_name}{uuid.uuid4()}".encode()).hexdigest()  # type: ignore

        # Keep the original file extension if it exists
        original_extension = Path(file_name).suffix
        hashed_filename = f"{random_hash}{original_extension}"

        file_path = Path(FILES_DIR) / hashed_filename

        temp_file_path = file_path.with_suffix(file_path.suffix + '.tmp')

        if temp_file_path.exists():
            os.remove(temp_file_path)

        temp_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                async for chunk in request.stream():
                    await f.write(chunk)

            os.rename(temp_file_path, file_path)

            file_item = FileItem(
                file_name=hashed_filename,
                file_name_orig=file_name,
                user_id=user_id,
                created_at=datetime.now(),
            )

            if not await self._files_repository.create_file(file_item):
                raise Exception("Failed to save file information to database")

            return JSONResponse(
                content={
                    "status": "success",
                    "file_name": file_name,
                    "stored_as": hashed_filename
                },
                media_type="application/json"
            )

        except Exception as e:
            exception("exception")
            if temp_file_path.exists():
                os.remove(temp_file_path)

            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )
