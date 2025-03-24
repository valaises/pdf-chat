import json
from typing import List, Any, Dict

from chat_toolbox.chat_models import ChatMessage, ChatMessageTool, ChatTool
from chat_toolbox.tool_usage.tool_abstract import Tool, build_tool_call
from chat_toolbox.tool_usage.tool_utils import messages_since_last_user_message, get_unanswered_tool_calls


TOOLS: List[Tool] = [

]


def execute_tools(ctx: Any, messages: List[ChatMessage]) -> List[ChatMessageTool]:
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

        ok, msgs = tool.validate_tool_call_args(tool_call, args)
        tool_res_messages.extend(msgs)

        if not ok:
            continue

        _ok, msgs = tool.execute(ctx, tool_call, args)
        tool_res_messages.extend(msgs)

    return tool_res_messages


def get_tools_list() -> List[ChatTool]:
    return [t.as_chat_tool() for t in TOOLS]


def get_system_prompts() -> Dict[str, str]:
    return {
        t.name: t.system_prompt() for t in TOOLS
    }
