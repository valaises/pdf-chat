import json

from typing import Dict, Any, List, Optional

from core.logger import warn
from core.repositories.repo_files import FileItem
from core.tools.tool_context import ToolContext
from core.tools.tool_utils import build_tool_call
from chat_tools.tool_usage.tool_abstract import Tool, ToolProps
from openai_wrappers.types import ToolCall, ChatMessage
from chat_tools.chat_models import (ChatTool,
    ChatToolFunction, ChatToolParameters,
    ChatToolParameterProperty
)
from vectors.search.search import vector_search_chat_messages


SYSTEM = """TOOL: search_in_doc
        Use this tool to retrieve relevant information from a document
        
        When search_in_doc is called, vector search is performed: search(query, chunks_of_doc), and most relevant chunks are retrieved
        
        Call when any condition bellow is met:
        * User asks a question about a document or related to a document
        * User directly asks to search in the document
        
        Prerequisites:
        * call list_documents to ensure the document exists and is ready
        
        Arguments:
        * query: descriptive, brief query used in search
        * document_name: document name retrieved from the list_documents tool
        * filters: optional dictionary to refine search results e.g. {"key": "value"}, retrieved from the list_documents tool
          filters should only be used when user directly asks to use them.
          
        search_in_doc produces:
        * pieces of text containing relevant information that will help you to answer user's question in a format:
          {"text": "text", "id": "pid-12345678"}
          
        When using text for composing answer, obligatory refer to id after piece of generated answer.
        Example:
          Sky is blue because of a phenomenon called Rayleigh scattering. [pid-12345678] Answer continues...
          Would you like me to help yo with anything else?
"""


class ToolSearchInFile(Tool):
    """
    A tool for searching within documents using vector search.

    This tool allows users to search for relevant information within a specified document
    by performing vector search on document chunks. It retrieves the most relevant
    sections of the document based on the provided query.

    The tool requires a document name and search query, and optionally accepts filters
    to refine search results.

    Before using this tool, the document must exist and be indexed in the vector store.
    The list_documents tool should be called first to verify document availability.

    Returns:
        A list of relevant document sections with metadata such as page numbers,
        section names, and highlight boxes when available.
    """
    @property
    def name(self) -> str:
        return "search_in_doc"

    def validate_tool_call_args(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        document_name = args.get("document_name")
        query = args.get("query")
        filters = args.get("filters")

        if not document_name or not query:
            return False, [
                build_tool_call(
                    f"Error validating tool {self.name}: Required args are either empty or missing. Non-empty 'document_name' and 'query' are required.",
                    tool_call
                )
            ]

        if filters:
            if not isinstance(filters, dict):
                return False, [
                    build_tool_call(
                        f"Error validating tool {self.name}: 'filters' should be a dictionary if provided.",
                        tool_call
                    )
                ]
            keys = list(filters.keys())
            if keys != ["section_name"]:
                return False, [
                    build_tool_call(
                        f"Error validating tool {self.name}: 'filters' keys could only contain 'section_name'.",
                        tool_call
                    )
                ]

        return True, []

    async def execute(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        document_name = args.get("document_name")
        query = args.get("query")

        try:
            files = await ctx.files_repository.get_files_by_filter(
                "user_id=?",
                (ctx.user_id,)
            )
        except Exception as e:
            err = f"Error while executing tool {self.name}: couldn't get user's documents: {str(e)}"
            warn(err)
            return False, [
                build_tool_call(
                    err, tool_call
                )
            ]

        document: Optional[FileItem] = next((f for f in files if f.file_name_orig == document_name), None)
        if not document:
            return False, [
                build_tool_call(
                    f"Error while executing tool {self.name}: document {document_name} not found."
                    f"All documents:\n{json.dumps([f.model_dump() for f in files])}",
                    tool_call
                )
            ]

        return await vector_search_chat_messages(ctx, self.name, document, query, tool_call)

    def as_chat_tool(self) -> ChatTool:
        return ChatTool(
            type="function",
            function=ChatToolFunction(
                name="search_in_doc",
                description="Searches for relevant information within a specified document.",
                parameters=ChatToolParameters(
                    type="object",
                    properties={
                        "document_name": ChatToolParameterProperty(
                            type="string",
                            description="The name of the document to search within.",
                            enum=[]
                        ),
                        "query": ChatToolParameterProperty(
                            type="string",
                            description="The search query to find relevant information.",
                            enum=[]
                        ),
                        # todo: add filters
                    },
                    required=["document_name", "query"]
                )
            )
        )

    def props(self):
        return ToolProps(
            tool_name=self.name,
            system_prompt=SYSTEM,
            depends_on=["list_documents"]
        )
