import asyncio
import uuid
from pathlib import Path
from typing import List, Iterator

from openai import OpenAI

from core.globals import ASSETS_DIR
from core.logger import init_logger, info
from core.repositories.repo_files import FileItem
from eval.extract_and_process import extract_and_process_files
from eval.globals import EVAL_USER_ID, PROCESSING_STRATEGY, SAVE_STRATEGY, DB_EVAL_DIR

from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


__all__ = []


def main():
    init_logger(True)
    info("Logger initialized")

    # assets/eval/*.pdf
    eval_files: Iterator[Path] = (
        f for f in (ASSETS_DIR / "eval").iterdir()
        if f.is_file() and f.suffix == ".pdf"
    )
    eval_files: List[FileItem] = [
        FileItem(
            file_name=f"file_{uuid.uuid4().hex[:24]}",
            file_name_orig=f.name,
            user_id=EVAL_USER_ID
        )
        for f in eval_files
    ]

    redis_repository = None
    milvus_repository = None

    if PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "redis":
        redis_repository = RedisRepository()
        redis_repository.connect()

    elif PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "milvus":
        milvus_repository = MilvusRepository(DB_EVAL_DIR / "milvus.db")

    loop = asyncio.new_event_loop()
    client = OpenAI()

    file_paragraphs_dict = extract_and_process_files(
        loop,
        client,
        redis_repository,
        milvus_repository,
        eval_files,
    )


if __name__ == "__main__":
    main()
