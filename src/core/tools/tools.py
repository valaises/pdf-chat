import json
from typing import List, Dict

from chat_toolbox.chat_models import ChatMessage, ChatMessageTool, ChatTool
from chat_toolbox.tool_usage.tool_abstract import Tool, build_tool_call, ToolProps
from chat_toolbox.tool_usage.tool_utils import messages_since_last_user_message, get_unanswered_tool_calls
from core.tools.tool_context import ToolContext
from core.tools.tool_list_files import ToolListFiles
from core.tools.tool_search_in_file import SearchInFile


__all__ = ['execute_tools', 'get_tools_list', 'get_tool_props']


TOOLS: List[Tool] = [
    ToolListFiles(),
    SearchInFile(),
]


async def execute_tools(ctx: ToolContext, messages: List[ChatMessage]) -> List[ChatMessageTool]:
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
    return [t.as_chat_tool() for t in TOOLS]


def get_tool_props() -> Dict[str, ToolProps]:
    return {
        t.name: t.props() for t in TOOLS
    }
