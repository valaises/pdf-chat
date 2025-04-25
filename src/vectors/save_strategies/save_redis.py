import json
from typing import List

from core.processing.local_fs.models import WorkerContext
from core.processing.p_models import ParagraphVectorData
from core.repositories.repo_files import FileItem
from vectors.repositories.repo_redis import VectorItem


def save_vectors_to_redis(
        ctx: WorkerContext,
        file: FileItem,
        paragraph_vectors: List[ParagraphVectorData]
) -> None:
    """
    Save paragraph vectors to Redis.

    Args:
        ctx: Worker context containing Redis repository
        file: file these vectors belong to
        paragraph_vectors: List of paragraph vector data to save
    """
    if not paragraph_vectors:
        raise Exception("No paragraph vectors provided")

    index_name = file.file_name

    # Check if index exists, if not create it
    if not ctx.repo_redis.index_exists(index_name):
        # Get dimension from the first vector
        if not paragraph_vectors[0].embedding:
            raise Exception("No embedding found in paragraph_vectors[0].embedding")

        dimension = len(paragraph_vectors[0].embedding)
        ctx.repo_redis.create_vector_index(index_name, dimension)

    vectors = [
        VectorItem(
            pv.paragraph_id,
            pv.embedding,
            {
                "paragraph_box": json.dumps(pv.paragraph_box),
                "text": pv.text,
                "page_n": pv.page_n,
                "idx": pv.idx,
                "file_name": file.file_name,
                "file_name_orig": file.file_name_orig,
            }
        )
        for pv in paragraph_vectors
        if pv.embedding
    ]

    if not vectors:
        raise Exception(f"No vectors to save for file: {file.file_name}")

    ctx.repo_redis.add_vectors(index_name, vectors)
    ctx.repo_redis.trigger_save()
