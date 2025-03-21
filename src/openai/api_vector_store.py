import aiohttp
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Optional

from core.logger import error


@dataclass
class VectorStoreCreate:
    name: str
    file_ids: List[str]


@dataclass
class VectorStoreSearch:
    vector_store_id: str
    query: str
    max_num_results: Optional[int] = 10


async def vector_store_create(data: VectorStoreCreate):
    client = OpenAI()

    vector_store = client.beta.vector_stores.create(
        name=data.name,
        file_ids=data.file_ids,
    )

    return vector_store


async def vector_stores_list():
    client = OpenAI()
    vector_stores = client.beta.vector_stores.list()
    return vector_stores


async def vector_store_retrieve(vector_store_id: str):
    client = OpenAI()
    vector_store = client.beta.vector_stores.retrieve(vector_store_id=vector_store_id)
    return vector_store


async def vector_store_search(data: VectorStoreSearch):
    api_key = OpenAI().api_key

    url = f"https://api.openai.com/v1/vector_stores/{data.vector_store_id}/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": data.query,
        "max_num_results": data.max_num_results
    }

    if hasattr(data, 'filters') and data.filters:
        payload["filters"] = data.filters

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result
            else:
                error_text = await resp.text()
                text = f"Error searching vector store: {error_text}"
                error(text)
                raise Exception(text)
