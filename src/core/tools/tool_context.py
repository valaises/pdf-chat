from dataclasses import dataclass

from core.repositories.repo_files import FilesRepository


@dataclass
class ToolContext:
    user_id: int
    files_repository: FilesRepository
