import json
from typing import Dict, Any, List

from chat_tools.chat_models import ToolCall, ChatMessage, ChatTool, ChatToolFunction, ChatToolParameters
from chat_tools.tool_usage.tool_abstract import Tool, build_tool_call, ToolProps
from core.tools.tool_context import ToolContext


SYSTEM = """TOOL: list_documents
        Use this tool to list the documents that can be used for context.
        
        Call when any condition bellow is met:
        * User implies desire to chat about a document -- call this tool to verify it exists
        * Check the processing_status of a document, re-check if document.processing_status was previously not complete
        
        Output format:
        [{
            file_name: str
            processing_status: str,
            available_filters: [...]
        }]
                
        processing_status variants:
        * "": when document is uploaded, awaiting to get extracted
        * extracted: when document is parsed, awaiting to be processed
        * processing: processing started
        * incomplete: processed partially
        * complete: complete, indexed, and ready to operate
        * Error: {error_text}: failed to be extracted or complete
"""


class ToolListFiles(Tool):
    @property
    def name(self) -> str:
        return "list_documents"

    def validate_tool_call_args(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        return True, []

    async def execute(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        try:
            files = ctx.files_repository.get_files_by_filter_sync(
                "user_id=?",
                (ctx.user_id,)
            )

            return True, [
                build_tool_call(
                    json.dumps([{
                        "file_name": f.file_name_orig,
                        "processing_status": f.processing_status,
                        "available_filters": ["section_name"]
                    }
                        for f in files], indent=2),
                    tool_call
                )
            ]
        except Exception as e:
            return False, [
                build_tool_call(
                    f"Error while executing tool {self.name}: {str(e)}",
                    tool_call
                )
            ]

    def as_chat_tool(self) -> ChatTool:
        return ChatTool(
            type="function",
            function=ChatToolFunction(
                name="list_documents",
                description="Lists all available documents for context with their processing status",
                parameters=ChatToolParameters(
                    type="object",
                    properties={},
                    required=[]
                )
            )
        )

    def props(self):
        return ToolProps(
            tool_name=self.name,
            system_prompt=SYSTEM,
        )
