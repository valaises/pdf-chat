import json
from typing import List

from fastapi import APIRouter, Response
from pydantic import BaseModel
from core.repositories.repo_files import FilesRepository
from core.tools.tools import get_tools_list, execute_tools, get_system_prompts

from chat_toolbox.chat_models import ChatMessage


class ToolsExecutePost(BaseModel):
    messages: List[ChatMessage]


class MCPLikeRouter(APIRouter):
    def __init__(
            self,
            files_repository: FilesRepository,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._files_repository = files_repository

        self.add_api_route("/v1/tools", self._tools, methods=["GET"])
        self.add_api_route("/v1/tools-execute", self._tools, methods=["POST"])
        self.add_api_route("/v1/system-prompts", self._system_prompts, methods=["GET"])

    async def _tools(self):
        content = {
            "tools": [t.model_dump() for t in get_tools_list()]
        }
        return Response(
            content=json.dumps(content, indent=2),
            media_type="application/json"
        )

    async def _execute_tools(self, post: ToolsExecutePost):
        tool_res_messages = execute_tools(None, post.messages)
        content = {
            "tool_res_messages": [msg.model_dump() for msg in tool_res_messages]
        }
        return Response(
            content=json.dumps(content, indent=2),
            media_type="application/json"
        )

    async def _system_prompts(self):
        # as tools is standardized under OpenAI format, we should not include any additional fields,
        # => system_prompts moved into its own endpoint
        content = {
            "system_prompts": get_system_prompts()
        }
        return Response(
            content=json.dumps(content, indent=2),
            media_type="application/json"
        )
