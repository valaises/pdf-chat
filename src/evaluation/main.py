import asyncio
import uuid
from pathlib import Path
from typing import List, Iterator, Tuple

import aiohttp
from openai import OpenAI

from core.globals import ASSETS_DIR
from core.logger import init_logger, info
from core.repositories.repo_files import FileItem
from evaluation.eval import evaluate_model_outputs
from evaluation.extract_and_process import extract_and_process_files
from evaluation.globals import EVAL_USER_ID, PROCESSING_STRATEGY, SAVE_STRATEGY, DB_EVAL_DIR
from evaluation.golden_answers import produce_golden_answers
from evaluation.questions import load_combined_questions
from processing.p_models import ParagraphData

from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


__all__ = []


def main():
    init_logger(False)
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
    http_session = aiohttp.ClientSession(loop=loop)

    questions = load_combined_questions(
        ASSETS_DIR / "eval" / "questions_str.json",
        ASSETS_DIR / "eval" / "questions_split.json",
    )

    questions = questions[:5] # todo: for test runs only

    try:
        file_paragraphs_dict = extract_and_process_files(
            loop,
            client,
            redis_repository,
            milvus_repository,
            eval_files,
        )

        file_paragraphs: List[Tuple[FileItem, List[ParagraphData]]] = [
            (f, file_paragraphs_dict[f.file_name_orig]) for f in eval_files
        ]
        golden_answers = produce_golden_answers(loop, http_session, file_paragraphs, questions)
        info(golden_answers)

        eval_results = evaluate_model_outputs(loop, http_session, questions, golden_answers)
        info(eval_results)

    finally:
        loop.run_until_complete(http_session.close())
        loop.close()


if __name__ == "__main__":
    main()
