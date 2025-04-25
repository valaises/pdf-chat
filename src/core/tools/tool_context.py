from dataclasses import dataclass
from typing import Optional

import aiohttp
from openai import OpenAI

from core.repositories.repo_files import FilesRepository
from core.repositories.repo_redis import RedisRepository


@dataclass
class ToolContext:
    http_session: aiohttp.ClientSession
    user_id: int
    files_repository: FilesRepository
    redis_repository: Optional[RedisRepository] = None
    openai: Optional[OpenAI] = None
