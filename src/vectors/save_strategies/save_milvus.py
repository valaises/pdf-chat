from typing import List

import numpy as np
from more_itertools.more import chunked

from core.logger import info
from core.processing.local_fs.models import WorkerContext
from core.processing.p_models import ParagraphVectorData
from core.repositories.repo_files import FileItem
from vectors.repositories.repo_milvus import VectorItem, collection_from_file_name


def save_vectors_to_milvus(
        ctx: WorkerContext,
        file: FileItem,
        paragraph_vectors: List[ParagraphVectorData]
) -> None:
    if not paragraph_vectors:
        raise Exception("No paragraph vectors provided")

    col_name = collection_from_file_name(file.file_name)
    collection = None
    try:
        collection = ctx.repo_milvus.collection_info(col_name)
    except Exception:
        pass

    if not collection:
        if not paragraph_vectors[0].embedding:
            raise Exception("No embedding found in paragraph_vectors[0].embedding")

        dimension = len(paragraph_vectors[0].embedding)

        ctx.repo_milvus.prepare_collection(col_name, dimension)

    vectors = (
        VectorItem(
            par_id=pv.paragraph_id,
            vector=np.array(pv.embedding),
            text=pv.text,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            idx=pv.idx,
            page_n=pv.page_n,
            paragraph_box=list(pv.paragraph_box),
        )
        for pv in paragraph_vectors
    )

    if not vectors:
        raise Exception(f"No vectors to save for file: {file.file_name}")

    for batch in chunked(vectors, 128):
        ctx.repo_milvus.insert(col_name, batch)
