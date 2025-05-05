import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Iterator, Tuple, Dict

import aiohttp
from openai import OpenAI

from core.globals import ASSETS_DIR, DB_DIR
from core.logger import init_logger, info
from core.repositories.repo_files import FileItem, FilesRepository
from core.tools.tool_context import ToolContext
from evaluation.stage2_answers.ans_golden import produce_golden_answers
from evaluation.stage3_evaluation.eval_collect_metrics import collect_eval_metrics
from evaluation.stage3_evaluation.llm_judge import evaluate_model_outputs
from evaluation.stage1_extraction.extract_and_process import extract_and_process_files
from evaluation.globals import EVAL_USER_ID, PROCESSING_STRATEGY, SAVE_STRATEGY, DB_EVAL_DIR
from evaluation.questions import load_combined_questions
from evaluation.stage2_answers.ans_rag import produce_rag_answers
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
            user_id=EVAL_USER_ID,
            processing_status="completed",
        )
        for f in eval_files
    ]

    files_repository = FilesRepository(DB_DIR / "files.db")
    files_repository.delete_user_files_sync(EVAL_USER_ID)
    for file in eval_files:
        assert files_repository.create_file_sync(file), f"Failed to create record of file={file} in DB"

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

    # questions = questions[:3] # todo: for test runs only

    tool_context = ToolContext(
        http_session=http_session,
        user_id=EVAL_USER_ID,
        files_repository=files_repository,
        redis_repository=redis_repository,
        milvus_repository=milvus_repository,
        openai=client,
    )

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

        eval_golden = evaluate_model_outputs(loop, http_session, questions, golden_answers)

        rag_results = produce_rag_answers(tool_context, loop, eval_files, questions)
        rag_answers: Dict[str, Dict[int, str]] = {
            file_name: {
                k: v[-1].content for k, v in fn_results.items()
            } for file_name, fn_results in rag_results.items()
        }

        eval_pred = evaluate_model_outputs(loop, http_session, questions, rag_answers)
        info(f"{eval_pred=}")

        t0 = time.time()
        eval_metrics = collect_eval_metrics(
            eval_golden, eval_pred
        )
        info(f"{eval_metrics=}")
        info(f"collect_eval_metrics: {time.time() - t0:.2f}s")


    finally:
        files_repository.delete_user_files_sync(EVAL_USER_ID)
        loop.run_until_complete(http_session.close())
        loop.close()


if __name__ == "__main__":
    main()
