from dataclasses import dataclass
from typing import Optional

from processing import p_models
from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


@dataclass
class WorkerContext(p_models.WorkerContext):
    repo_redis: Optional[RedisRepository] = None
    repo_milvus: Optional[MilvusRepository] = None
