import json

from typing import List

from openai_wrappers.utils import messages_since_last_user_message, get_unanswered_tool_calls
from openai_wrappers.types import ChatMessage, ChatMessageTool, ChatTool

from core.tools.tool_abstract import build_tool_call, Tool, ToolProps
from core.tools.tool_context import ToolContext
from core.tools.tool_list_files import ToolListFiles
from core.tools.tool_search_in_file import ToolSearchInFile


__all__ = ['execute_tools', 'get_tools_list', 'get_tool_props']


TOOLS: List[Tool] = [
    ToolListFiles(),
    ToolSearchInFile(),
]


async def execute_tools(ctx: ToolContext, messages: List[ChatMessage]) -> List[ChatMessageTool]:
    """
    Execute tool calls found in the conversation messages.

    This function processes unanswered tool calls in the most recent conversation segment
    (since the last user message). For each tool call, it:
    1. Finds the corresponding tool implementation
    2. Parses and validates the arguments
    3. Executes the tool if validation passes

    Args:
        ctx (ToolContext): The context object providing environment and state for tool execution
        messages (List[ChatMessage]): The full conversation history

    Returns:
        List[ChatMessageTool]: A list of tool response messages to be added to the conversation

    Note:
        - Only processes messages since the last user message
        - Handles JSON parsing errors in tool arguments
        - Validates tool arguments before execution
        - Collects all tool response messages, including error messages
    """
    messages = messages_since_last_user_message(messages)

    tool_res_messages = []
    for tool_call in get_unanswered_tool_calls(messages):
        tool = next((t for t in TOOLS if t.name == tool_call.function.name), None)
        if not tool:
            continue

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            tool_res_messages.append(build_tool_call(
                f"Error: invalid JSON in arguments: {tool_call.function.arguments}", tool_call
            ))
            continue

        ok, msgs = tool.validate_tool_call_args(ctx, tool_call, args)
        tool_res_messages.extend(msgs)

        if not ok:
            continue

        _ok, msgs = await tool.execute(ctx, tool_call, args)
        tool_res_messages.extend(msgs)

    return tool_res_messages


def get_tools_list() -> List[ChatTool]:
    """
    Get a list of all available tools in their ChatTool format.

    This function converts all registered tools to the ChatTool format,
    which is suitable for passing to chat models that support tool calling.

    Returns:
        List[ChatTool]: A list of all available tools in ChatTool format
    """
    return [t.as_chat_tool() for t in TOOLS]


def get_tool_props() -> List[ToolProps]:
    """
    Get the properties of all available tools.

    This function retrieves the ToolProps for each registered tool,
    which includes metadata like the tool name and system prompt.
    These properties are used for tool registration and documentation.

    Returns:
        List[ToolProps]: A list of property objects for all available tools
    """
    return [t.props() for t in TOOLS]
