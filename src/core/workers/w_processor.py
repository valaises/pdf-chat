import threading

from core.globals import PROCESSING_STRATEGY
from core.processing.openai_fs.worker import p_openai_fs_worker
from core.processing.local_fs.worker import p_local_fs_worker
from core.repositories.repo_files import FilesRepository
from core.workers.w_abstract import Worker


def spawn_worker(
        files_repository: FilesRepository,
) -> Worker:
    stop_event = threading.Event()

    if PROCESSING_STRATEGY == "openai_fs":
        worker = p_openai_fs_worker
    elif PROCESSING_STRATEGY == "local_fs":
        worker = p_local_fs_worker
    else:
        raise ValueError(f"Unknown P_STRATEGY: {PROCESSING_STRATEGY}")

    worker_thread = threading.Thread(
        target=worker,
        args=(stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return Worker("worker_processor", worker_thread, stop_event)
