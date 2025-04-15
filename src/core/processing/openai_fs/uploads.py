from typing import List, Tuple, Any

import ujson as json

from core.logger import info
from core.processing.p_models import WorkerContext, ParagraphData
from core.processing.p_utils import generate_paragraph_id
from openai_wrappers.api_files import async_file_upload, FileUpload
from openai_wrappers.api_vector_store import async_vector_store_file_create, VectorStoreFileCreate


async def upload_paragraph_if_needed(
        ctx: WorkerContext,
        para: ParagraphData,
        filename: str,
        openai_files_list: List[Any]
) -> Tuple[Any, float]:
    """Upload a paragraph as a file if it doesn't already exist in OpenAI.

    This function checks if a file with the given filename already exists in the OpenAI files list.
    If not, it uploads the paragraph text as a new file to OpenAI's assistants API.

    Args:
        ctx: Worker context containing the client and event loop
        para: Paragraph data containing the text to upload
        filename: Name to use for the file in OpenAI
        openai_files_list: List of existing OpenAI files to check against

    Returns:
        A tuple containing:
        - The file data object from OpenAI
        - The time taken for the upload in seconds (0 if no upload was needed)
    """
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
    """Add a file to the OpenAI vector store if it's not already present.

    This function checks if the given file is already in the vector store by comparing file IDs.
    If not present, it adds the file to the vector store with metadata attributes from the paragraph.

    Args:
        ctx: Worker context containing the client and event loop
        para: Paragraph data containing metadata to store as attributes
        filename: Name of the file for logging purposes
        file_data: The file object returned from OpenAI's file upload
        vector_store: The vector store object where the file should be added
        vector_store_files: List of existing files in the vector store to check against

    Returns:
        The time taken for adding to the vector store in seconds (0 if no addition was needed)

    Note:
        The function attaches metadata like page number, paragraph ID, and bounding box
        coordinates as attributes in the vector store for improved searchability.
    """
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
