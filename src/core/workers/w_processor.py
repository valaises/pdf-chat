import asyncio
import hashlib
import json
import threading

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set

from pydantic import BaseModel
from openai import OpenAI

from core.globals import FILES_DIR
from core.logger import info, error, exception
from core.repositories.repo_files import FilesRepository
from core.workers.w_abstract import Worker
from core.workers.w_utils import jsonl_reader
from openai_wrappers.api_files import files_list, FileUpload, file_upload, file_delete
from openai_wrappers.api_vector_store import (
    vector_stores_list, VectorStoreCreate, vector_store_create,
    vector_store_files_list, VectorStoreFilesList,
    vector_store_file_create, VectorStoreFileCreate,
    vector_store_file_delete
)


class ParagraphData(BaseModel):
    page_n: int
    section_number: Optional[str] = None
    paragraph_text: str
    paragraph_box: Tuple[float, float, float, float]


def generate_hashed_filename(
        base_name: str,
        content: str,
        extension: str
) -> str:
    """
    Generate a filename using a hash of the base_name and content to ensure uniqueness.

    Args:
        base_name: The base name of the file
        content: The content to hash (typically paragraph text)
        extension: The file extension including the dot

    Returns:
        A unique filename with format {base_name}_{hash}{extension}
    """
    # noinspection PyTypeChecker
    content_hash = hashlib.md5((base_name + content).encode()).hexdigest()[:16]
    return f"{base_name}_{content_hash}{extension}"


def worker(
        stop_event: threading.Event,
        files_repository: FilesRepository
) -> None:
    client = OpenAI()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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
                stop_event.wait(30)
                continue

            openai_resources = get_openai_resources(client)
            if not openai_resources:
                continue

            openai_files_list, openai_vector_stores = openai_resources

            for file in process_files:
                process_single_file(client, file, files_repository, openai_files_list, openai_vector_stores)

            stop_event.wait(5)
    finally:
        loop.close()


def get_files_to_process(files_repository: FilesRepository) -> List[Any]:
    """Get files that need processing from the repository."""
    return files_repository.get_files_by_filter_sync(
        "processing_status IN (?, ?)",
        ("extracted", "incomplete")
    )


def get_openai_resources(client: OpenAI) -> Optional[Tuple[List[Any], List[Any]]]:
    """Retrieve necessary OpenAI resources for processing."""
    try:
        openai_files_list = files_list(client)
        openai_vector_stores = vector_stores_list(client)
        return openai_files_list, openai_vector_stores
    except Exception as e:
        error(f"Error retrieving OpenAI files_list or vector_stores_list: {str(e)}")
        return None


def process_single_file(
        client: OpenAI,
        file: Any,
        files_repository: FilesRepository,
        openai_files_list: List[Any],
        openai_vector_stores: List[Any]
) -> None:
    """Process a single file through the entire pipeline."""
    file.processing_status = "processing"
    info(f"Processing file: {file.file_name_orig}")

    jsonl_file_path = get_jsonl_file_path(file)
    if not jsonl_file_path or not jsonl_file_path.is_file():
        mark_file_as_error(file, files_repository, "Error: jsonl file not found on disk")
        return

    vector_store = ensure_vector_store_exists(client, file, files_repository, openai_vector_stores)
    if not vector_store:
        return

    vector_store_files = get_vector_store_files(client, vector_store)
    if vector_store_files is None:
        return

    # Generate expected filenames from disk content
    expected_filenames = generate_expected_filenames(
        jsonl_file_path,
        Path(file.file_name_orig).stem,
        Path(file.file_name).suffix
    )

    # Process paragraphs
    process_file_paragraphs(
        client,
        file,
        jsonl_file_path,
        openai_files_list,
        vector_store,
        vector_store_files,
        files_repository
    )

    # Update vector store files after new files are added
    vector_store_files = get_vector_store_files(client, vector_store)
    if vector_store_files is None:
        return

    # Clean up files that exist in vector store but not in disk content
    cleanup_orphaned_files(client, vector_store, vector_store_files, expected_filenames)

    if file.processing_status != "incomplete":
        file.processing_status = "complete"
        files_repository.update_file_sync(file.file_name, file)

    info(f"{file.processing_status}; File {file.file_name}")


def get_jsonl_file_path(file: Any) -> Path:
    """Get the path to the JSONL file for a given file."""
    file_path: Path = FILES_DIR.joinpath(file.file_name)
    return file_path.with_suffix('.jsonl')


def mark_file_as_error(file: Any, files_repository: FilesRepository, error_message: str) -> None:
    """Mark a file as having an error in processing."""
    file.processing_status = error_message
    files_repository.update_file_sync(file.file_name, file)


def ensure_vector_store_exists(
        client: OpenAI,
        file: Any,
        files_repository: FilesRepository,
        openai_vector_stores: List[Any]
) -> Optional[Any]:
    """Ensure a vector store exists for the file, creating one if needed."""
    vector_store = next((s for s in openai_vector_stores if s.name == file.file_name), None)
    if not vector_store:
        payload = VectorStoreCreate(
            name=file.file_name
        )
        try:
            vector_store = vector_store_create(client, payload)
            info(f"Vector store created for {file.file_name}")
        except Exception as e:
            error(f"Error while creating vector store for {file.file_name}: {str(e)}")
            return None

    if file.vector_store_id != vector_store.id:
        file.vector_store_id = vector_store.id
        files_repository.update_file_sync(file.file_name, file)

    return vector_store


def get_vector_store_files(client: OpenAI, vector_store: Any) -> Optional[List[Any]]:
    """Get the files associated with a vector store."""
    try:
        return vector_store_files_list(
            client,
            VectorStoreFilesList(
                vector_store_id=vector_store.id
            )
        )
    except Exception as e:
        error(f"Error retrieving vector store files for {vector_store.name}: {str(e)}")
        return None


def generate_expected_filenames(
        jsonl_file_path: Path,
        base_name: str,
        extension: str
) -> Set[str]:
    """Generate the set of expected filenames based on disk content."""
    expected_filenames = set()

    for data_dict in jsonl_reader(jsonl_file_path):
        try:
            para = ParagraphData(**data_dict)
            filename = generate_hashed_filename(base_name, para.paragraph_text, extension)
            expected_filenames.add(filename)
        except Exception as e:
            error(f"Error generating expected filename: {str(e)}")

    return expected_filenames


def cleanup_orphaned_files(
        client: OpenAI,
        vector_store: Any,
        vector_store_files: List[Any],
        expected_filenames: Set[str]
) -> None:
    """Delete files that exist in vector store but not in disk content."""
    for vs_file in vector_store_files:
        # Get the filename from the OpenAI file object
        try:
            # Find the corresponding file in OpenAI files
            openai_file = client.files.retrieve(file_id=vs_file.id)
            filename = openai_file.filename

            if filename not in expected_filenames:
                info(f"Deleting orphaned file from vector store: {filename} (ID: {vs_file.id})")

                try:
                    # First delete from vector store
                    vector_store_file_delete(client, vector_store.id, vs_file.id)
                    info(f"Successfully removed file {filename} from vector store {vector_store.id}")

                    # Then delete the file itself
                    file_delete(client, vs_file.id)
                    info(f"Successfully deleted file {filename} (ID: {vs_file.id})")
                except Exception as e:
                    error(f"Error deleting orphaned file {filename} (ID: {vs_file.id}): {str(e)}")
        except Exception as e:
            error(f"Error retrieving file info for ID {vs_file.id}: {str(e)}")


def process_file_paragraphs(
        client: OpenAI,
        file: Any,
        jsonl_file_path: Path,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any],
        files_repository: FilesRepository
) -> None:
    """Process all paragraphs in a file."""
    base_name = Path(file.file_name_orig).stem
    extension = Path(file.file_name).suffix

    for (idx, data_dict) in enumerate(jsonl_reader(jsonl_file_path)):
        try:
            process_paragraph(
                client,
                data_dict,
                base_name,
                extension,
                openai_files_list,
                vector_store,
                vector_store_files
            )
        except Exception as e:
            exception(f"Error uploading file:\n{file}\n{str(e)}")
            file.processing_status = "incomplete"
            files_repository.update_file_sync(file.file_name, file)
            break


def process_paragraph(
        client: OpenAI,
        data_dict: Dict[str, Any],
        base_name: str,
        extension: str,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any]
) -> None:
    """Process a single paragraph from a file."""
    para = ParagraphData(**data_dict)
    filename = generate_hashed_filename(base_name, para.paragraph_text, extension)

    file_data = upload_paragraph_if_needed(client, para, filename, openai_files_list)
    if not file_data:
        return

    add_to_vector_store_if_needed(
        client,
        para,
        filename,
        file_data,
        vector_store,
        vector_store_files,
    )


def upload_paragraph_if_needed(
        client: OpenAI,
        para: ParagraphData,
        filename: str,
        openai_files_list: List[Any]
) -> Optional[Any]:
    """Upload a paragraph as a file if it doesn't already exist."""
    file_data = next((f for f in openai_files_list if f.filename == filename), None)
    if not file_data:
        info(f"Uploading file: {filename}")
        try:
            file2upload = FileUpload(
                para.paragraph_text.encode(),
                filename,
                "assistants"
            )
            file_data = file_upload(client, file2upload)
            info(f"OK -- file uploaded: {filename} with file_id: {file_data.id}")
        except Exception as e:
            error(f"Error uploading file {filename}: {str(e)}")
            return None
    else:
        info(f"File {filename} already exists with file_id: {file_data.id}")

    return file_data


def add_to_vector_store_if_needed(
        client: OpenAI,
        para: ParagraphData,
        filename: str,
        file_data: Any,
        vector_store: Any,
        vector_store_files: List[Any],
) -> None:
    """Add a file to the vector store if it's not already there."""
    chunking_strategy = None  # todo: implement

    vector_store_file = next((f for f in vector_store_files if f.id == file_data.id), None)
    if not vector_store_file:
        info(f"Adding File {filename} to vector store {vector_store.id}")

        attributes = {
            "page_n": para.page_n,
            "paragraph_box": json.dumps(list(para.paragraph_box)),
        }
        if para.section_number:
            attributes["section_number"] = para.section_number

        try:
            vector_store_file = vector_store_file_create(
                client,
                VectorStoreFileCreate(
                    vector_store_id=vector_store.id,
                    file_id=file_data.id,
                    attributes=attributes,
                    chunking_strategy=chunking_strategy,
                )
            )
            info(f"OK -- File {filename} added to vector store {vector_store.id} with id: {vector_store_file.id}")
        except Exception as e:
            error(f"Error adding file {filename} to vector store: {str(e)}")
    else:
        info(f"File {filename} already exists in vector store {vector_store.id}")


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
