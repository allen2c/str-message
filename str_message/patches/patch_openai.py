import typing

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic.fields import FieldInfo


def patch_openai():
    """Apply all OpenAI SDK patches for third-party API compatibility.
    Patches models to support Groq/Gemini-specific fields missing in OpenAI spec.
    """
    # Patch ChatCompletionMessage
    patch_chat_cmpl_message_reasoning()

    # Patch ChatCompletion
    patch_chat_cmpl_service_tier()

    # Patch ChoiceDeltaToolCall for Gemini compatibility
    patch_choice_delta_tool_call_index()


def patch_chat_cmpl_service_tier():
    """Add 'on_demand' service tier option for Groq API compatibility.
    Groq uses 'on_demand' tier which is not in OpenAI's original spec.
    """
    NewServiceTierType = typing.Optional[
        typing.Literal["auto", "default", "flex", "scale", "priority", "on_demand"]
    ]
    ChatCompletion.__annotations__["service_tier"] = NewServiceTierType
    ChatCompletion.model_fields["service_tier"] = FieldInfo(
        annotation=NewServiceTierType, default=None  # type: ignore
    )
    ChatCompletion.model_rebuild(force=True)


def patch_chat_cmpl_message_reasoning():
    """Add 'reasoning' field to ChatCompletionMessage for Groq reasoning models.
    Groq exposes reasoning text in responses, but OpenAI SDK doesn't support it.
    """
    ChatCompletionMessage.__annotations__["reasoning"] = typing.Optional[str]
    ChatCompletionMessage.model_fields["reasoning"] = FieldInfo(
        annotation=typing.Optional[str], default=None  # type: ignore
    )
    ChatCompletionMessage.model_rebuild(force=True)


def patch_choice_delta_tool_call_index():
    """Patch ChoiceDeltaToolCall.index to have default value 0.
    Gemini streaming API returns None for index, violating OpenAI spec.
    """
    ChoiceDeltaToolCall.model_fields["index"] = FieldInfo(
        annotation=int, default=0  # type: ignore
    )
    ChoiceDeltaToolCall.model_rebuild(force=True)
