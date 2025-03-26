from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel


type ChatMessage = Union[ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool]


class ChatToolParameterProperty(BaseModel):
    type: str
    description: str
    enum: List[str] = None


class ChatToolParameters(BaseModel):
    type: str
    properties: Dict[str, ChatToolParameterProperty]
    required: List[str]
    additionalProperties: bool = False


class ChatToolFunction(BaseModel):
    name: str
    description: str
    parameters: ChatToolParameters
    strict: bool = True


class ChatTool(BaseModel):
    type: str
    function: ChatToolFunction


class ChatMessageContentItemText(BaseModel):
    text: str
    type: str


class ChatMessageContentItemImage(BaseModel):
    image_url: str
    type: str


class ChatMessageContentItemAudio(BaseModel):
    input_audio: str
    type: Literal["input_audio"]


class ChatMessageContentItemFile(BaseModel):
    file: str
    type: Literal["file"]


class ChatMessageBase(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, Union[
        ChatMessageContentItemText,
        ChatMessageContentItemImage,
        ChatMessageContentItemAudio,
        ChatMessageContentItemFile
    ]]


class ChatMessageSystem(ChatMessageBase):
    role: Literal["system", "developer"]
    name: Optional[str] = None


class ChatMessageUser(ChatMessageBase):
    role: Literal["user"]
    name: Optional[str] = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessageAssistant(ChatMessageBase):
    role: Literal["assistant"]
    refusal: Optional[str] = None
    name: Optional[str] = None
    audio: Optional[Any] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatMessageTool(ChatMessageBase):
    role: Literal["tool"]
    tool_call_id: Optional[str] = None


class ChatFunctionParameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class ChatFunctionParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, ChatFunctionParameter]
    required: Optional[List[str]] = None


class ChatFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: ChatFunctionParameters
