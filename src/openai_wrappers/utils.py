from typing import List, Iterator


from openai_wrappers.types import ChatMessage, ChatMessageUser, ChatMessageTool, ChatMessageAssistant, ToolCall


def messages_since_last_user_message(messages: List[ChatMessage]):
    messages_since_last_user_msg = []
    for message in reversed(messages):
        if isinstance(message, ChatMessageUser):
            break
        messages_since_last_user_msg.insert(0, message)
    return messages_since_last_user_msg


def get_unanswered_tool_calls(messages: List[ChatMessage]) -> Iterator[ToolCall]:
    # Get all tool call IDs from responses
    tool_messages: List[ChatMessageTool] = [
        m for m in messages if isinstance(m, ChatMessageTool)
    ]
    answered_tool_call_ids = {
        m.tool_call_id for m in tool_messages if m.tool_call_id
    }

    # yield all unanswered tool calls
    for m in messages:
        if isinstance(m, ChatMessageAssistant) and m.tool_calls:
            for tool_call in m.tool_calls:
                if tool_call.id not in answered_tool_call_ids:
                    yield tool_call


def chat_message_readable(message: ChatMessage) -> str:
    """
    Convert a ChatMessage to a readable string format.

    Only includes role, content, tool_calls, and tool_call_id fields.
    Tool calls are formatted as "tool_call: tool_name, parameters: parameters".

    Args:
        message: A ChatMessage object

    Returns:
        A readable string representation of the message
    """
    parts = []

    parts.append(f"ROLE: {message.role}")

    if isinstance(message.content, str):
        parts.append(f"content:\n{message.content}")
    elif isinstance(message.content, list):
        content_texts = []
        for item in message.content:
            if hasattr(item, "text") and item.type == "text":
                content_texts.append(item.text)
        if content_texts:
            parts.append(f"content: {' '.join(content_texts)}")

    # Add tool_calls if present (for assistant messages)
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"tool_call: {tool_call.function.name}, parameters: {tool_call.function.arguments}")

    # Add tool_call_id if present (for tool messages)
    if hasattr(message, "tool_call_id") and message.tool_call_id:
        parts.append(f"tool_call_id: {message.tool_call_id}")

    return "\n".join(parts)
