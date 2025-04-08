import time
import threading

from pathlib import Path
from typing import Iterator


from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from xattr import xattr

from core.logger import info, error, warn
from core.repositories.repo_files import FilesRepository, FileItem
from core.workers.w_abstract import Worker


__all__ = ["spawn_worker"]

FILE_RULE = "*.[pP][dD][fF]"


def add_file_to_db_if_ok(file_path: Path, files_repository: FilesRepository) -> None:
    files = files_repository.get_files_by_filter_sync("file_name = ?", (file_path.name,))
    if len(files):
        info(f"File {file_path.name} already in DB. SKIP")
        return

    try:
        file_attrs = xattr(str(file_path))
        user_id = (file_attrs.get('user.user_id') or b"").decode('utf-8', errors='ignore')
        file_name_orig = (file_attrs.get('user.file_name_orig') or b"").decode('utf-8', errors='ignore')
    except Exception as e:
        warn(f"Failed to parse {file_path.name}'s metadata: {e}")
        return

    info(f"USERID: {user_id}, file_name_orig: {file_name_orig}")

    if not user_id:
        warn(f"File {file_path.name} does not contain a user.userid in metadata. SKIP")
        return

    if not file_name_orig:
        warn(f"File {file_path.name} does not contain a user.file_name_orig in metadata. SKIP")
        return

    file_item = FileItem(
        file_name=file_path.name,
        file_name_orig=file_name_orig,
        user_id=int(user_id),
    )

    resp = files_repository.create_file_sync(file_item)

    if not resp:
        error(f"Failed to create file's {file_item.file_name} record in DB")
        return

    info(f"Created file {file_item.file_name} record in DB successfully")


def scan_existing_files(directory: Path) -> Iterator[Path]:
    for file_path in directory.glob(FILE_RULE): # non-recursive
        if file_path.is_file():
            yield file_path


class EventHandler(FileSystemEventHandler):
    def __init__(self, files_repository):
        super().__init__()
        self.files_repository = files_repository

    def on_created(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        if path.match(FILE_RULE):
            # todo: potential problem: when adding a file in progress, watchdog may loose incoming files
            add_file_to_db_if_ok(path, self.files_repository)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events by removing the corresponding record from the database."""
        path = Path(event.src_path)
        if path.match(FILE_RULE):
            file_name = path.name
            if self.files_repository.delete_file_sync(file_name):
                info(f"Removed file {file_name} from database after deletion from disk")
            else:
                warn(f"Failed to remove file {file_name} from database or file not found in database")

def worker(target_dir: Path, stop_event: threading.Event, files_repository):
    event_handler = EventHandler(files_repository)
    observer = Observer()
    observer.schedule(event_handler, str(target_dir), recursive=False)
    observer.start()
    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


def spawn_worker(
        target_dir: Path,
        files_repository: FilesRepository,
) -> Worker:
    existing_files = list(scan_existing_files(target_dir))
    existing_file_names = [file_path.name for file_path in existing_files]

    # Clean up database records for files that no longer exist on disk
    removed_count = files_repository.cleanup_missing_files_sync(existing_file_names)
    if removed_count > 0:
        info(f"Cleaned up {removed_count} database records for files that no longer exist on disk")

    # Add existing files to the database
    for file_path in existing_files:
        add_file_to_db_if_ok(file_path, files_repository)

    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker,
        args=(target_dir, stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return Worker("worker_watchdog", worker_thread, stop_event)
