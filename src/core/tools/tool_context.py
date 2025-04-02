from dataclasses import dataclass

import aiohttp

from core.repositories.repo_files import FilesRepository


@dataclass
class ToolContext:
    http_session: aiohttp.ClientSession
    user_id: int
    files_repository: FilesRepository
