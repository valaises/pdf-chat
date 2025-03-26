import aiohttp
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Any, Literal

from core.logger import error


@dataclass
class VectorStoreFileCreate:
    vector_store_id: str
    file_id: str
    attributes: Optional[Dict[str, Union[str, bool, int, float]]] = None
    chunking_strategy: Optional[Dict] = None


@dataclass
class VectorStoreCreate:
    name: str
    file_ids: Optional[List[str]] = None


@dataclass
class VectorStoreFilesList:
    vector_store_id: str
    limit: int = 100


@dataclass
class VectorStoreSearchFilterComparison:
    type: Literal["eq", "ne", "lt", "lte", "gt", "gte"]
    property: str
    value: str


@dataclass
class VectorStoreSearchFilterCompound:
    type: Literal["and", "or"]
    filters: List[VectorStoreSearchFilterComparison]


@dataclass
class VectorStoreSearch:
    vector_store_id: str
    query: str
    filters: Optional[List[VectorStoreSearchFilterCompound | VectorStoreSearchFilterComparison]] = None
    max_num_results: Optional[int] = 10


@dataclass
class VectorStoreSearchRespItemContentItem:
    type: str
    text: str


@dataclass
class VectorStoreSearchRespItem:
    file_id: str
    filename: str
    score: float
    content: List[VectorStoreSearchRespItemContentItem]
    attributes: Optional[Dict[str, Any]] = None


def vector_store_create(data: VectorStoreCreate, client: Optional[OpenAI] = None):
    client = client or OpenAI()

    vector_store = client.vector_stores.create(
        name=data.name,
        file_ids=data.file_ids,
    )

    return vector_store


def vector_stores_list(client: Optional[OpenAI] = None):
    client = client or OpenAI()

    vector_stores = client.vector_stores.list()
    return vector_stores


def vector_store_retrieve(vector_store_id: str, client: Optional[OpenAI] = None):
    client = client or OpenAI()

    vector_store = client.vector_stores.retrieve(vector_store_id=vector_store_id)
    return vector_store


def vector_store_file_create(data: VectorStoreFileCreate, client: Optional[OpenAI] = None):
    client = client or OpenAI()

    params = {
        "file_id": data.file_id
    }

    if data.attributes:
        params["attributes"] = data.attributes

    if data.chunking_strategy:
        params["chunking_strategy"] = data.chunking_strategy

    vector_store_file = client.vector_stores.files.create(
        vector_store_id=data.vector_store_id,
        **params
    )

    return vector_store_file


def vector_store_files_list(data: VectorStoreFilesList, client: Optional[OpenAI] = None):
    client = client or OpenAI()

    files = list(client.vector_stores.files.list(
        vector_store_id=data.vector_store_id,
        limit=data.limit
    ))

    if len(files) == data.limit:
        while True:
            try:
                batch = list(client.vector_stores.files.list(
                    vector_store_id=data.vector_store_id,
                    limit=data.limit,
                    after=(files[-1] or {}).get("id")
                ))
                files.extend(batch)
                if len(batch) != data.limit:
                    break
            except Exception as e:
                error(f"Error fetching additional vector store files: {e}")
                break

    return files


async def vector_store_search(data: VectorStoreSearch) -> List[VectorStoreSearchRespItem]:
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

                try:
                    # Convert raw results to VectorStoreSearchRespItem objects
                    items = []
                    for item in result["data"]:
                        content_items = []
                        for content in item['content']:
                            content_items.append(VectorStoreSearchRespItemContentItem(
                                type=content['type'],
                                text=content['text']
                            ))

                        items.append(VectorStoreSearchRespItem(
                            file_id=item['file_id'],
                            filename=item['filename'],
                            score=item['score'],
                            content=content_items,
                            attributes=item.get('attributes')
                        ))

                    return items
                except KeyError as e:
                    error_msg = f"Failed to convert search results: Missing required field {e}"
                    error(error_msg)
                    raise ValueError(error_msg)
                except Exception as e:
                    error_msg = f"Failed to convert search results: {str(e)}"
                    error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_text = await resp.text()
                text = f"Error searching vector store: {error_text}"
                error(text)
                raise Exception(text)
