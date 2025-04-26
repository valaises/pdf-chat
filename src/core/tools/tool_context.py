from dataclasses import dataclass
from typing import Optional

import aiohttp
from openai import OpenAI

from core.repositories.repo_files import FilesRepository
from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


@dataclass
class ToolContext:
    http_session: aiohttp.ClientSession
    user_id: int
    files_repository: FilesRepository
    redis_repository: Optional[RedisRepository] = None
    milvus_repository: Optional[MilvusRepository] = None

    openai: Optional[OpenAI] = None
