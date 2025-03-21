from typing import Optional

from fastapi import APIRouter, Response
from pydantic import BaseModel

from core.globals import FILES_DIR
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

    async def _create_file(self, file: FileItem):
        if not FILES_DIR.joinpath(file.file_name).is_file():
            return Response(status_code=404, content={"message": "file not found"})

        resp = await self._files_repository.create_file(file)
        if resp:
            return Response(status_code=201, content={"message": "added record of file in DB"})
        else:
            return Response(status_code=400, content={"message": "failed to add record of file in DB"})

    async def _delete_file(self, delete_file_post: DeleteFilePost):
        resp = await self._files_repository.delete_file(delete_file_post.file_name)
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

