import json
import asyncio
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

from pydantic import BaseModel
from openai import OpenAI

from core.globals import FILES_DIR
from core.logger import info, error, exception, warn
from core.repositories.repo_files import FilesRepository, FileItem
from core.telemetry import TeleWriter, TelemetryScope, TeleWProcessor, TeleItemStatus
from core.workers.w_abstract import Worker
from core.workers.w_utils import jsonl_reader, generate_paragraph_id, \
    generate_vector_store_file_name, generate_hashed_filename
from openai_wrappers.api_files import files_list, FileUpload, async_file_upload
from openai_wrappers.api_vector_store import (
    vector_stores_list, VectorStoreCreate, vector_store_create,
    vector_store_files_list, VectorStoreFilesList,
    VectorStoreFileCreate, async_vector_store_file_create
)


class ParagraphData(BaseModel):
    page_n: int
    section_number: Optional[str] = None
    paragraph_text: str
    paragraph_box: Tuple[float, float, float, float]


@dataclass
class WorkerContext:
    client: OpenAI
    loop: asyncio.AbstractEventLoop
    tele: TeleWriter
    files_repository: FilesRepository


def worker(
        stop_event: threading.Event,
        files_repository: FilesRepository
) -> None:
    """
    Main worker function that processes files in a continuous loop until stopped.

    This worker handles the entire file processing pipeline:
    1. Initializes OpenAI client and event loop
    2. Resets any files stuck in "processing" status to "incomplete"
    3. Continuously polls for files that need processing
    4. Retrieves necessary OpenAI resources (files list and vector stores)
    5. Processes each file through the complete pipeline

    The worker will pause between iterations and can be gracefully stopped
    using the provided stop_event.

    Args:
        stop_event: Threading event used to signal the worker to stop
        files_repository: Repository for accessing and updating file data

    Returns:
        None
    """
    client = OpenAI()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tele = TeleWriter(TelemetryScope.W_PROCESSOR)

    ctx = WorkerContext(
        client=client,
        loop=loop,
        tele=tele,
        files_repository=files_repository
    )

    processing_files = files_repository.get_files_by_filter_sync(
        "processing_status IN (?)",
        ("processing",)
    )
    for file in processing_files:
        file.processing_status = "incomplete"
        files_repository.update_file_sync(file.file_name, file)

    try:
        while not stop_event.is_set():
            process_files = get_files_to_process(files_repository)

            if not process_files:
                stop_event.wait(3)
                continue

            t0 = time.time()
            try:
                openai_files_list, openai_vector_stores = get_openai_resources(client)
            except Exception as e:
                error(f"Error retrieving OpenAI files_list or vector_stores_list: {str(e)}")
                TeleWProcessor(
                    event="get_openai_resources",
                    status=TeleItemStatus.FAILURE,
                    error_message=str(e),
                    error_recoverable=True,
                    duration_seconds=time.time() - t0,
                ).write(ctx.tele)
                continue
            else:
                TeleWProcessor(
                    event="get_openai_resources",
                    status=TeleItemStatus.SUCCESS,
                    duration_seconds = time.time() - t0,
                ).write(ctx.tele)

            for file in process_files:
                process_single_file(
                    ctx,
                    file,
                    openai_files_list,
                    openai_vector_stores
                )

            stop_event.wait(1)
    finally:
        loop.close()


def get_files_to_process(files_repository: FilesRepository) -> List[FileItem]:
    """Get files that need processing from the repository."""
    return files_repository.get_files_by_filter_sync(
        "processing_status IN (?, ?)",
        ("extracted", "incomplete")
    )


def get_openai_resources(client: OpenAI) -> Tuple[List[Any], List[Any]]:
    """Retrieve necessary OpenAI resources for processing."""
    openai_files_list = files_list(client)
    openai_vector_stores = vector_stores_list(client)
    return openai_files_list, openai_vector_stores


def process_single_file(
        ctx: WorkerContext,
        file: FileItem,
        openai_files_list: List[Any],
        openai_vector_stores: List[Any]
) -> None:
    """Process a single file through the entire pipeline."""
    info(f"Processing file: {file.file_name_orig} STATUS={file.processing_status}")
    file.processing_status = "processing"

    ts_process_single_file = time.time()

    jsonl_file_path = get_jsonl_file_path(file)
    if not jsonl_file_path or not jsonl_file_path.is_file():
        error_message = "Error: jsonl file not found on disk"
        error(error_message)
        mark_file_as_error(file, ctx.files_repository, error_message)
        TeleWProcessor(
            event="get_jsonl_file_path",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=False,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
        ).write(ctx.tele)
        return

    t0 = time.time()
    try:
        vector_store = ensure_vector_store_exists(ctx, file, ctx.files_repository, openai_vector_stores)
    except Exception as e:
        error_message = f"Error while creating vector store for {file.file_name}: {str(e)}"
        mark_file_as_error(file, ctx.files_repository, error_message)
        error(error_message)
        TeleWProcessor(
            event="ensure_vector_store_exists",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=False,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
        return

    t0 = time.time()
    try:
        vector_store_files = get_vector_store_files(ctx, vector_store)
    except Exception as e:
        error_message = f"Error retrieving vector store files for {vector_store.name}: {str(e)}"
        mark_file_as_error(file, ctx.files_repository, error_message)
        error(error_message)
        TeleWProcessor(
            error_message=error_message,
            error_recoverable=False,
            event="get_vector_store_files",
            status=TeleItemStatus.FAILURE,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
        return

    # Process paragraphs
    t0 = time.time()
    try:
        # Use ctx.loop to run the coroutine
        ctx.loop.run_until_complete(process_file_paragraphs(
            ctx,
            file,
            jsonl_file_path,
            openai_files_list,
            vector_store,
            vector_store_files
        ))
    except Exception as e:
        error_message = f"Error processing paragraphs:\n{file}\n{str(e)}"
        exception(error_message)
        file.processing_status = "incomplete"
        ctx.files_repository.update_file_sync(file.file_name, file)
        TeleWProcessor(
            error_message=error_message,
            error_recoverable=True,
            event="process_file_paragraphs",
            status=TeleItemStatus.FAILURE,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
    else:
        TeleWProcessor(
            event="process_file_paragraphs",
            status=TeleItemStatus.SUCCESS,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)

    if file.processing_status != "incomplete":
        file.processing_status = "complete"
        ctx.files_repository.update_file_sync(file.file_name, file)

    TeleWProcessor(
        event="process_file_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        vector_store=vector_store.name,
        attributes={
            "processing_status": file.processing_status
        },
        duration_seconds=time.time() - ts_process_single_file
    ).write(ctx.tele)

    info(f"{file.processing_status}; File {file.file_name}")


def get_jsonl_file_path(file: FileItem) -> Path:
    """Get the path to the JSONL file for a given file."""
    file_path: Path = FILES_DIR.joinpath(file.file_name)
    return file_path.with_suffix('.jsonl')


def mark_file_as_error(file: FileItem, files_repository: FilesRepository, error_message: str) -> None:
    """Mark a file as having an error in processing."""
    file.processing_status = error_message
    files_repository.update_file_sync(file.file_name, file)


def ensure_vector_store_exists(
        ctx: WorkerContext,
        file: FileItem,
        files_repository: FilesRepository,
        openai_vector_stores: List[Any]
) -> Any:
    """Ensure a vector store exists for the file, creating one if needed."""
    vs_file_name = generate_vector_store_file_name(file)
    vector_store = next((s for s in openai_vector_stores if s.name == vs_file_name), None)
    if not vector_store:
        payload = VectorStoreCreate(
            name=vs_file_name
        )
        vector_store = vector_store_create(ctx.client, payload)
        info(f"Vector store created for {file.file_name}")

    if file.vector_store_id != vector_store.id:
        file.vector_store_id = vector_store.id
        files_repository.update_file_sync(file.file_name, file)

    return vector_store


def get_vector_store_files(ctx: WorkerContext, vector_store: Any) -> List[Any]:
    """Get the files associated with a vector store."""
    return vector_store_files_list(
        ctx.client,
        VectorStoreFilesList(
            vector_store_id=vector_store.id
        )
    )


async def process_file_paragraphs(
        ctx: WorkerContext,
        file: FileItem,
        jsonl_file_path: Path,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any],
) -> None:
    """Process all paragraphs in a file with controlled concurrency."""
    base_name = Path(file.file_name_orig).stem
    extension = Path(file.file_name).suffix

    t0 = ctx.loop.time()
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10)  # Process up to 10 paragraphs concurrently

    async def process_with_semaphore(data_dict):
        async with semaphore:
            return await process_paragraph(
                ctx,
                file,
                data_dict,
                base_name,
                extension,
                openai_files_list,
                vector_store,
                vector_store_files
            )

    # Create tasks for all paragraphs
    tasks = []
    for data_dict in jsonl_reader(jsonl_file_path):
        tasks.append(process_with_semaphore(data_dict))

    # Process all paragraphs with controlled concurrency
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    # Calculate statistics
    # if 0 -> cached value was used
    # if None -> failed making a request
    add_file_times = [r[0] for r in results if r[0]]
    add_file_to_vs_times = [r[1] for r in results if r[1]]

    stats = {
        "add_file": {
            "items_from_cache": sum(1 for r in results if r[0] is None),
            "total_items": len(results),
            "min_time": None,
            "max_time": None,
            "avg_time": None,
            "med_time": None,
            # Total processing time: Track the overall time taken for the entire batch of paragraphs
            "total_time": sum(add_file_times) if add_file_times else 0,
            # Success rate: Percentage of paragraphs successfully processed
            "success_rate": len([r[0] for r in results if r[0] is not None]) / len(tasks) if tasks else 0,
            # Throughput: Number of paragraphs processed per second
            "throughput": len(add_file_times) / (sum(add_file_times) if add_file_times else 1),
            "p95_time": float(np.percentile(add_file_times, 95)) if add_file_times else None,
            "p99_time": float(np.percentile(add_file_times, 99)) if add_file_times else None,
            # Error counts: Number of errors encountered during processing
            "error_count": len(tasks) - len(results),
        },
        "add_file_to_vs": {
            "items_from_cache": sum(1 for r in results if r[1] is None),
            "total_items": len(results),
            "min_time": None,
            "max_time": None,
            "avg_time": None,
            "med_time": None,
            "total_time": sum(add_file_to_vs_times) if add_file_to_vs_times else 0,
            "success_rate": len([r[1] for r in results if r[1] is not None]) / len(tasks) if tasks else 0,
            "throughput": len(add_file_to_vs_times) / (sum(add_file_to_vs_times) if add_file_to_vs_times else 1),
            "p95_time": float(np.percentile(add_file_to_vs_times, 95)) if add_file_to_vs_times else None,
            "p99_time": float(np.percentile(add_file_to_vs_times, 99)) if add_file_to_vs_times else None,
            "error_count": len(tasks) - len(results),
        }
    }

    if add_file_times:
        stats["add_file"]["min_time"] = min(add_file_times)
        stats["add_file"]["max_time"] = max(add_file_times)
        try:
            stats["add_file"]["avg_time"] = sum(add_file_times) / len(add_file_times)
        except Exception as e:
            warn(e)
        try:
            stats["add_file"]["med_time"] = sorted(add_file_times)[len(add_file_times) // 2]
        except Exception as e:
            warn(e)

    if add_file_to_vs_times:
        stats["add_file_to_vs"]["min_time"] = min(add_file_to_vs_times)
        stats["add_file_to_vs"]["max_time"] = max(add_file_to_vs_times)
        try:
            stats["add_file_to_vs"]["avg_time"] = sum(add_file_to_vs_times) / len(add_file_to_vs_times)
        except Exception as e:
            warn(e)
        try:
            stats["add_file_to_vs"]["med_time"] = sorted(add_file_to_vs_times)[len(add_file_to_vs_times) // 2]
        except Exception as e:
            warn(e)

    TeleWProcessor(
        event="process_paragraphs_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        vector_store=vector_store.name,
        attributes={
            "stats": stats
        },
        duration_seconds=ctx.loop.time() - t0
    ).write(ctx.tele)


async def process_paragraph(
        ctx: WorkerContext,
        file: FileItem,
        data_dict: Dict[str, Any],
        base_name: str,
        extension: str,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any]
) -> Optional[
    Tuple[Optional[float], Optional[float]]
]:
    """
    Process a single paragraph from a document file.

    This function handles the complete paragraph processing pipeline:
    1. Converts raw dictionary data to a structured ParagraphData object
    2. Generates a unique filename for the paragraph
    3. Uploads the paragraph to OpenAI (with 5s timeout)
    4. Adds the paragraph to the vector store (with 5s timeout)

    Both operations are performed with error handling and telemetry reporting.
    If any step fails, the file is marked as incomplete for future retry.

    Args:
        ctx: Worker context containing client, loop, telemetry and repository
        file: The file item being processed
        data_dict: Dictionary containing paragraph data from the JSONL file
        base_name: Base name of the original file
        extension: File extension
        openai_files_list: List of existing OpenAI files
        vector_store: The vector store to add paragraphs to
        vector_store_files: List of files already in the vector store

    Returns:
        Optional tuple containing upload duration and vector store addition duration,
        or None if any operation failed
    """
    para = ParagraphData(**data_dict)
    filename = generate_hashed_filename(base_name, para.paragraph_text, extension)

    t0 = ctx.loop.time()
    try:
        file_data, dur_upload_para = await asyncio.wait_for(
            upload_paragraph_if_needed(ctx, para, filename, openai_files_list),
            timeout=5.0
        )
    except Exception as e:
        error_message = f"Error uploading paragraph: {str(e)}"
        if isinstance(e, asyncio.TimeoutError):
            error_message = "Timeout uploading paragraph (exceeded 5s)"
        error(error_message)

        if file.processing_status != "incomplete":
            file.processing_status = "incomplete"
            await ctx.files_repository.update_file(file.file_name, file)

        TeleWProcessor(
            event="upload_paragraph",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=True,
            user_id=file.user_id,
            attributes={
                "filename": filename,
                "paragraph_length": len(para.paragraph_text),
                "page_n": para.page_n,
                "section_number": para.section_number
            },
            duration_seconds=ctx.loop.time() - t0
        ).write(ctx.tele)
        return

    t0 = ctx.loop.time()
    try:
        dur_add_to_vs = await asyncio.wait_for(
            add_to_vector_store_if_needed(
                ctx,
                para,
                filename,
                file_data,
                vector_store,
                vector_store_files,
            ),
            timeout=5.0
        )
    except Exception as e:
        error_message = f"Error adding paragraph to vector store: {str(e)}"
        if isinstance(e, asyncio.TimeoutError):
            error_message = "Timeout adding paragraph to vector store (exceeded 5s)"
        error(error_message)

        if file.processing_status != "incomplete":
            file.processing_status = "incomplete"
            await ctx.files_repository.update_file(file.file_name, file)

        TeleWProcessor(
            event="add_to_vector_store",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=True,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=ctx.loop.time() - t0
        ).write(ctx.tele)
        return dur_upload_para, None

    return dur_upload_para, dur_add_to_vs


async def upload_paragraph_if_needed(
        ctx: WorkerContext,
        para: ParagraphData,
        filename: str,
        openai_files_list: List[Any]
) -> Tuple[Any, float]:
    """Upload a paragraph as a file if it doesn't already exist."""
    file_data = next((f for f in openai_files_list if f.filename == filename), None)
    if not file_data:
        info(f"Uploading file: {filename}")
        file2upload = FileUpload(
            para.paragraph_text.encode(),
            filename,
            "assistants"
        )
        t0 = ctx.loop.time()
        file_data = await async_file_upload(ctx.client, file2upload)
        info(f"OK -- file uploaded: {filename} with file_id: {file_data.id}")
        return file_data, ctx.loop.time() - t0

    info(f"File {filename} already exists with file_id: {file_data.id}")
    return file_data, 0


async def add_to_vector_store_if_needed(
        ctx: WorkerContext,
        para: ParagraphData,
        filename: str,
        file_data: Any,
        vector_store: Any,
        vector_store_files: List[Any],
) -> float:
    """Add a file to the vector store if it's not already there."""
    chunking_strategy = None  # todo: implement

    vector_store_file = next((f for f in vector_store_files if f.id == file_data.id), None)
    if vector_store_file:
        info(f"File {filename} already exists in vector store {vector_store.id}")
        return 0

    info(f"Adding File {filename} to vector store {vector_store.id}")

    paragraph_id = generate_paragraph_id(para.paragraph_text)

    attributes = {
        "page_n": para.page_n,
        "paragraph_id": paragraph_id,
        "paragraph_box": json.dumps(list(para.paragraph_box)),
    }

    if para.section_number:
        attributes["section_number"] = para.section_number

    t0 = ctx.loop.time()
    vector_store_file = await async_vector_store_file_create(
        ctx.client,
        VectorStoreFileCreate(
            vector_store_id=vector_store.id,
            file_id=file_data.id,
            attributes=attributes,
            chunking_strategy=chunking_strategy,
        )
    )
    info(f"OK -- File {filename} added to vector store {vector_store.id} with id: {vector_store_file.id}")
    return ctx.loop.time() - t0


def spawn_worker(
        files_repository: FilesRepository,
) -> Worker:
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker,
        args=(stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return Worker("worker_processor", worker_thread, stop_event)
