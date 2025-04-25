import asyncio
from pathlib import Path
from typing import List, Tuple, Set

import ujson as json
from more_itertools import chunked

from core.globals import SAVE_STRATEGY
from core.logger import error
from core.processing.local_fs.const import SEMAPHORE_LIMIT, EMBEDDING_BATCH_SIZE
from core.processing.local_fs.models import WorkerContext
from core.processing.local_fs.save_strategies import save_vectors_to_redis
from core.processing.p_models import ParagraphData, ParagraphVectorData
from core.processing.p_utils import (
    jsonl_reader,
    generate_paragraph_id, try_aggr_requests_stats
)
from core.repositories.repo_files import FileItem
from openai_wrappers.api_embeddings import async_create_embeddings
from telemetry.models import (
    TeleWProcessor, TeleItemStatus,
    RequestResult, RequestStatus
)


def count_tokens(text: str) -> int:
    return int((len(text) or 1) / 4)


def chunkify_text(text: str, min_tokens: int, max_tokens: int) -> List[str]:
    if count_tokens(text) < max_tokens:
        return [text]

    chunks = []

    buff = ""
    for line in text.splitlines(keepends=True):
        # assuming line is not nearly big as max_tokens
        new_buff = buff + line
        if count_tokens(new_buff) > min_tokens:
            chunks.append(buff)
            buff = ""
        else:
            buff = new_buff

    if buff:
        chunks.append(buff)

    return chunks


async def process_paragraphs_batch(
        ctx,
        file: FileItem,
        paragraphs: List[ParagraphData]
) -> Tuple[List[RequestResult], List[ParagraphVectorData]]:
    p_vecs = []
    for p in paragraphs:
        chunks = chunkify_text(p.paragraph_text, 256, 1024)
        for idx, chunk in enumerate(chunks):
            p_vecs.append(ParagraphVectorData(
                paragraph_box=p.paragraph_box,
                paragraph_id=p.paragraph_id,
                idx=idx,
                text=chunk,
                page_n=p.page_n,
            ))

    res_requests = []
    try:
        for p_vecs_batch in chunked(p_vecs, EMBEDDING_BATCH_SIZE):
            t0 = ctx.loop.time()
            try:
                res = await asyncio.wait_for(
                    async_create_embeddings(ctx.client, [p.text for p in p_vecs_batch]),
                    timeout=30.0
                )
            except Exception as e:
                error_message = f"Error fetching embedding: {str(e)}"
                if isinstance(e, asyncio.TimeoutError):
                    error_message = "Timeout fetching embedding (exceeded 30s)"
                error(error_message)

                res_requests.append(RequestResult(
                    "upload_paragraph",
                    status=RequestStatus.NOT_OK,
                    ts_created=t0,
                    duration_seconds=ctx.loop.time() - t0,
                    error_message=str(e),
                ))
                raise e

            embeddings = res.data
            if len(embeddings) != len(p_vecs_batch):
                raise Exception(f"Unexpected number of embeddings: {len(embeddings)} != {len(p_vecs_batch)}")

            embeddings.sort(key=lambda i: i.index)

            for p_vec, emb in zip(p_vecs_batch, embeddings):
                if not isinstance(emb.embedding, List):
                    raise Exception(f"Unexpected embedding type: {type(emb.embedding)}; Should be List[float]")
                p_vec.embedding = emb.embedding

    except Exception as e:
        error_message = f"Error occurred when fetching embeddings: {str(e)}"
        error(error_message)

        if file.processing_status != "incomplete":
            file.processing_status = "incomplete"
            await ctx.files_repository.update_file(file.file_name, file)

        return res_requests, []

    return res_requests, p_vecs


async def process_file_paragraphs(
        ctx: WorkerContext,
        file: FileItem,
        jsonl_file_path: Path,
        jsonl_vec: Path,
        processed_paragraphs: Set[str],
):
    t0 = ctx.loop.time()
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def process_paragraphs_batch_with_semaphore(
            _paragraphs: List[ParagraphData]
    ):
        async with semaphore:
            return await process_paragraphs_batch(ctx, file, _paragraphs)

    tasks = []
    for data_dict_batch in chunked(jsonl_reader(jsonl_file_path), EMBEDDING_BATCH_SIZE):
        paragraphs = [ParagraphData(**d) for d in data_dict_batch]
        for p in paragraphs:
            if not p.paragraph_id:
                p.paragraph_id = generate_paragraph_id(p.paragraph_text)

            if p.paragraph_id in processed_paragraphs:
                continue

        tasks.append(asyncio.create_task(
            process_paragraphs_batch_with_semaphore(paragraphs)
        ))


    results: List[Tuple[List[RequestResult], List[ParagraphVectorData]]] = await asyncio.gather(*tasks)


    req_results: List[RequestResult] = [r_i for r in results for r_i in r[0]]
    p_vecs: List[ParagraphVectorData] = [r_i for r in results for r_i in r[1]]

    stats = {
        "fetch_embeddings": try_aggr_requests_stats(req_results).to_dict()
    }

    # todo: add save_strategy time -t0 into telemetry
    match SAVE_STRATEGY:
        case "local":
            # todo: change to "a" when ready
            with jsonl_vec.open("w") as f:
                for p_vec in p_vecs:
                    f.write(json.dumps(p_vec.model_dump()) + "\n")

        case "redis":
            assert ctx.repo_redis is not None
            try:
                save_vectors_to_redis(ctx, file, p_vecs)
            except Exception as e:
                error_message = f"Error occurred when saving vectors to redis: {str(e)}"
                error(error_message)
                TeleWProcessor(
                    proc_strategy="local_fs",
                    event="process_paragraphs_done",
                    status=TeleItemStatus.FAILURE,
                    user_id=file.user_id,
                    file_name=file.file_name,
                    file_name_orig=file.file_name_orig,
                    attributes={
                        "stats": stats
                    },
                    error_message=error_message,
                ).write(ctx.tele)
                raise Exception(error_message)

        case _:
            raise Exception(f"Unexpected save strategy: {SAVE_STRATEGY}")


    TeleWProcessor(
        proc_strategy="local_fs",
        event="process_paragraphs_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        attributes={
            "stats": stats
        },
        duration_seconds=ctx.loop.time() - t0
    ).write(ctx.tele)
