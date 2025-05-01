import asyncio
from typing import List

from aiohttp import ClientSession

from core.logger import info
from evaluation.globals import CHAT_ENDPOINT_API_KEY, CHAT_ENDPOINT, CHAT_MODEL
from openai_wrappers.types import ChatMessage, ChatMessageUser


async def call_chat_completions_non_streaming(
        http_session: ClientSession,
        messages: List[ChatMessage]
):
    # Request payload with stream=False
    payload = {
        "model": CHAT_MODEL,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
        "temperature": 0.7,
    }

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


async def test_call_chat(http_session: ClientSession):
    messages = [
        ChatMessageUser(role="user", content="Hi there!"),
    ]
    result = await call_chat_completions_non_streaming(http_session, messages)
    info(result)


def call_chat(loop: asyncio.AbstractEventLoop, http_session: ClientSession):
    loop.run_until_complete(test_call_chat(http_session))
