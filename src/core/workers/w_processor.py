import asyncio
import threading

from pathlib import Path
from typing import Tuple

from core.globals import FILES_DIR
from core.logger import warn, info
from core.repositories.repo_files import FilesRepository


def worker(
        stop_event: threading.Event,
        stats_repository: FilesRepository
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not stop_event.is_set():
            process_files = stats_repository.get_files_by_filter_sync("processing_status = ?", ("extracted",))
            if not process_files:
                stop_event.wait(30)
                continue

            for file in process_files:
                info(f"Processing file: {file.file_name_orig}")
                file_path: Path = FILES_DIR.joinpath(file.file_name)
                jsonl_file_path = file_path.with_suffix('.jsonl')

                if not jsonl_file_path.is_file():
                    file.processing_status = f"Error: jsonl file not found on disk"
                    stats_repository.update_file_sync(file.file_name, file)
                    continue

                # here be code


            stop_event.wait(5)

    finally:
        loop.close()


def spawn_worker(
        files_repository: FilesRepository,
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker,
        args=(stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return stop_event, worker_thread
