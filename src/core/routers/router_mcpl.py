from typing import List, Optional

from chat_tools.tool_usage import ToolProps
from fastapi import APIRouter, status
from openai import OpenAI
from pydantic import BaseModel

import aiohttp

from chat_tools.chat_models import ChatTool

from core.repositories.repo_files import FilesRepository
from core.repositories.repo_redis import RedisRepository
from core.routers.schemas import  error_constructor, ErrorResponse
from core.tools.tool_context import ToolContext
from core.tools.tools import get_tools_list, execute_tools, get_tool_props
from openai_wrappers.types import ChatMessage


class ToolsExecutePost(BaseModel):
    user_id: int
    messages: List[ChatMessage]


class ToolsResponse(BaseModel):
    tools: List[ChatTool]


class ToolResMessagesResponse(BaseModel):
    tool_res_messages: List[ChatMessage]


class ToolPropsResponse(BaseModel):
    props: List[ToolProps]


class MCPLRouter(APIRouter):
    def __init__(
            self,
            http_session: aiohttp.ClientSession,
            files_repository: FilesRepository,
            redis_repository: Optional[RedisRepository] = None,
            openai: Optional[OpenAI] = None,
            *args, **kwargs
    ):
        kwargs["tags"] = ["MCPL"]
        super().__init__(*args, **kwargs)
        self.http_session = http_session
        self._files_repository = files_repository
        self._redis_repository = redis_repository
        self._openai = openai

        self.add_api_route(
            "/v1/tools",
            self._tools,
            methods=["GET"],
            response_model=ToolsResponse,
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "Successful response with a list of available tools",
                    "model": ToolsResponse
                },
                500: {
                    "description": "Failed to retrieve tools due to an internal server error",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "An error occurred while retrieving tools",
                                    "type": "mcpl_error",
                                    "code": "tools_retrieval_failed"
                                }
                            }
                        }
                    }
                }
            }
        )

        self.add_api_route(
            "/v1/tools-execute",
            self._execute_tools,
            methods=["POST"],
            response_model=ToolResMessagesResponse,
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "Successful execution of tools with resulting messages",
                    "model": ToolResMessagesResponse
                },
                400: {
                    "description": "Invalid request parameters",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "Invalid request parameters",
                                    "type": "mcpl_error",
                                    "code": "invalid_request"
                                }
                            }
                        }
                    }
                },
                500: {
                    "description": "Failed to execute tools due to an internal server error",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "An error occurred while executing tools",
                                    "type": "mcpl_error",
                                    "code": "tools_execution_failed"
                                }
                            }
                        }
                    }
                }
            }
        )

        self.add_api_route(
            "/v1/tools-props",
            self._tool_props,
            methods=["GET"],
            response_model=ToolPropsResponse,
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "Successful response with tool properties",
                    "model": ToolPropsResponse
                },
                500: {
                    "description": "Failed to retrieve tool properties due to an internal server error",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "An error occurred while retrieving tool properties",
                                    "type": "mcpl_error",
                                    "code": "tool_props_retrieval_failed"
                                }
                            }
                        }
                    }
                }
            }
        )

    async def _tools(self):
        """
        Get a list of available tools.

        This endpoint returns a list of all available tools that can be used with the system.

        Returns:
            A ToolsResponse object containing the list of available tools

        Raises:
        ```
        - 500: If an error occurs while retrieving tools
        ```
        """
        try:
            return ToolsResponse(
                tools=get_tools_list()
            )
        except Exception as e:
            return error_constructor(
                message=f"An error occurred while retrieving tools: {e}",
                error_type="mcpl_error",
                code="tools_retrieval_failed",
                status_code=500,
            )

    async def _execute_tools(self, post: ToolsExecutePost):
        """
        Execute pending tool calls from the provided messages.

        This endpoint processes the given messages, identifies unanswered tool calls
        since the last user message, and executes the appropriate tools with their
        specified arguments.

        Parameters:
        - **user_id**: The ID of the user making the request
        - **messages**: A list of chat messages containing tool calls to process

        Returns:
            A ToolResMessagesResponse object containing the tool response messages

        Raises:
        ```
        - 400: If the request parameters are invalid
        - 500: If an error occurs during tool execution
        ```
        """
        try:
            tool_context = ToolContext(
                self.http_session,
                post.user_id,
                self._files_repository,
                self._redis_repository,
                self._openai,
            )
            tool_res_messages = await execute_tools(tool_context, post.messages)
            return ToolResMessagesResponse(
                tool_res_messages=tool_res_messages
            )
        except ValueError as e:
            return error_constructor(
                message=f"Invalid request parameters: {e}",
                error_type="mcpl_error",
                code="invalid_request",
                status_code=400,
            )
        except Exception as e:
            return error_constructor(
                message=f"An error occurred while executing tools: {e}",
                error_type="mcpl_error",
                code="tools_execution_failed",
                status_code=500,
            )

    async def _tool_props(self):
        """
        Get properties for available tools.

        As tools are standardized under the OpenAI format, additional properties
        are provided through this separate endpoint.

        Returns:
            A ToolPropsResponse object containing the properties for available tools

        Raises:
        ```
        - 500: If an error occurs while retrieving tool properties
        ```
        """
        try:
            return ToolPropsResponse(
                props=[
                    props
                    for props in get_tool_props()
                    if props
                ]
            )
        except Exception as e:
            return error_constructor(
                message=f"An error occurred while retrieving tool properties: {e}",
                error_type="mcpl_error",
                code="tool_props_retrieval_failed",
                status_code=500,
            )
