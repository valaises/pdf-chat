from dataclasses import dataclass
from typing import Optional

from core.processing import p_models
from core.repositories.repo_redis import RedisRepository


@dataclass
class WorkerContext(p_models.WorkerContext):
    repo_redis: Optional[RedisRepository] = None
