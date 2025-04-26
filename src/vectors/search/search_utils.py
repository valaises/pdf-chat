import asyncio

from typing import Tuple, Any, Optional

import openai

from core.logger import info
from openai_wrappers.api_embeddings import async_create_embeddings


async def create_query_embedding(
        openai_client: openai.OpenAI,
        query: str,
        timeout: float = 5.0
) -> Tuple[bool, Any, Optional[str]]:
    """
    Create embedding for a search query.

    Args:
        openai_client: The OpenAI client instance
        query: The search query text
        timeout: Maximum time to wait for embedding creation

    Returns:
        Tuple containing:
        - Success flag (bool)
        - Embedding if successful, None otherwise
        - Error message if failed, None otherwise
    """
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    try:
        res = await asyncio.wait_for(
            async_create_embeddings(openai_client, [query]),
            timeout=timeout
        )
        embedding = res.data[0].embedding
        assert embedding is not None, "Embedding is empty"
        info(f"retrieved embedding in {loop.time() - t0:.2f}s")
        return True, embedding, None
    except Exception as e:
        err = f"Failed to fetch embeddings: {str(e)}"
        return False, None, err
