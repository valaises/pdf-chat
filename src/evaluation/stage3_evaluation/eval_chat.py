from typing import List, Dict

from aiohttp import ClientSession

from evaluation.globals import CHAT_ENDPOINT_API_KEY, CHAT_ENDPOINT
from openai_wrappers.types import ChatMessage


async def call_chat_completions_non_streaming(
        http_session: ClientSession,
        messages: List[ChatMessage],
        model: str,
        tools: List[Dict[str, str]] = None
):
    # Request payload with stream=False
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
        "temperature": 0.7,
    }

    if tools:
        payload["tools"] = tools

    # Headers including authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHAT_ENDPOINT_API_KEY}"
    }

    async with http_session.post(
            f"{CHAT_ENDPOINT}/chat/completions",
            json=payload,
            headers=headers
    ) as response:
        if response.status == 200:
            # Parse the JSON response
            result = await response.json()
            return result
        else:
            error_text = await response.text()
            return {
                "error": True,
                "status_code": response.status,
                "message": error_text
            }
