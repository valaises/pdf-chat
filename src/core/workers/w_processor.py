import asyncio
import threading

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from core.globals import FILES_DIR
from core.logger import info, warn, error, exception
from core.repositories.repo_files import FilesRepository
from core.workers.w_abstract import Worker
from core.workers.w_utils import jsonl_reader

from openai_wrappers.api_files import files_list, FileUpload, file_upload
from openai_wrappers.api_vector_store import (
    vector_stores_list, VectorStoreCreate, vector_store_create,
    vector_store_files_list, VectorStoreFilesList,
    vector_store_file_create, VectorStoreFileCreate
)


CHUNKING_STRATEGY = None # todo: implement


@dataclass
class Section:
    text: Optional[str]
    section_name: Optional[str]



def worker(
        stop_event: threading.Event,
        files_repository: FilesRepository
):
    client = OpenAI()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while not stop_event.is_set():
            process_files = files_repository.get_files_by_filter_sync(
                "processing_status IN (?, ?)",
                ("extracted", "incomplete")
            )

            if not process_files:
                stop_event.wait(30)
                continue

            try:
                openai_files_list = files_list()
                openai_vector_stores = vector_stores_list(client)
            except Exception as e:
                error(f"Error retrieving OpenAI files_list or vector_stores_list: {str(e)}")
                continue

            for file in process_files:
                file.processing_status = "processing"
                info(f"Processing file: {file.file_name_orig}")
                file_path: Path = FILES_DIR.joinpath(file.file_name)
                jsonl_file_path = file_path.with_suffix('.jsonl')

                if not jsonl_file_path.is_file():
                    file.processing_status = f"Error: jsonl file not found on disk"
                    files_repository.update_file_sync(file.file_name, file)
                    continue

                vector_store = next((s for s in openai_vector_stores if s.name == file.file_name), None)
                if not vector_store:
                    payload = VectorStoreCreate(file.file_name)
                    try:
                        vector_store = vector_store_create(payload, client)
                        info(f"Vector store created for {file.file_name}")
                    except Exception as e:
                        error(f"Error while creating vector store for {file.file_name}: {str(e)}")
                        continue

                if file.vector_store_id != vector_store.id:
                    file.vector_store_id = vector_store.id
                    files_repository.update_file_sync(file.file_name, file)

                try:
                    vector_store_files = vector_store_files_list(
                        VectorStoreFilesList(vector_store.id), client
                    )
                except Exception as e:
                    error(f"Error retrieving vector store files for {file.file_name}: {str(e)}")
                    continue

                for (idx, data_dict) in enumerate(jsonl_reader(jsonl_file_path)):
                    base_name = Path(file.file_name).stem
                    extension = Path(file.file_name).suffix
                    filename = f"{base_name}_{idx}{extension}"
                    try:
                        for k, v in data_dict.items():
                            section = Section(
                                section_name=k,
                                text=v,
                            )
                            if section.text is None or section.section_name is None:
                                warn(f"idx: {idx}; Skipping section {section.section_name}: section.text or section.name is empty")
                                continue

                            file_data = next((f for f in openai_files_list if f.filename == filename), None)
                            if not file_data:
                                info(f"Uploading file: {filename}")
                                file2upload = FileUpload(section.text.encode(), filename, "assistants")
                                file_data = file_upload(file2upload, client)
                                info(f"OK -- file uploaded: {filename} with file_id: {file_data.id}")
                            else:
                                info(f"File {filename} already exists with file_id: {file_data.id}")

                            vector_store_file = next((f for f in vector_store_files if f.id == file_data.id), None)
                            if not vector_store_file:
                                info(f"Adding File {filename} to vector store {vector_store.id}")
                                vector_store_file = vector_store_file_create(
                                    VectorStoreFileCreate(
                                        vector_store.id,
                                        file_data.id,
                                        {"section_name": section.section_name},
                                        chunking_strategy=CHUNKING_STRATEGY
                                    ), client
                                )
                                info(f"OK -- File {filename} added to vector store {vector_store.id} with id: {vector_store_file.id}")
                            else:
                                info(f"File {filename} already exists in vector store {vector_store.id}")

                    except Exception as e:
                        exception(f"Error uploading file:\n{file}\n{str(e)}")
                        file.processing_status = "incomplete"
                        files_repository.update_file_sync(file.file_name, file)

                if file.processing_status != "incomplete":
                    file.processing_status = "complete"
                    files_repository.update_file_sync(file.file_name, file)
                info(f"{file.processing_status}; File {file.file_name}")

            stop_event.wait(5)

    finally:
        loop.close()


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
