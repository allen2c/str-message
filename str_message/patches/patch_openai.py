import typing

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic.fields import FieldInfo


def patch_openai():
    # Patch ChatCompletionMessage
    patch_chat_cmpl_message_reasoning()

    # Patch ChatCompletion
    patch_chat_cmpl_service_tier()


def patch_chat_cmpl_service_tier():
    NewServiceTierType = typing.Optional[
        typing.Literal["auto", "default", "flex", "scale", "priority", "on_demand"]
    ]
    ChatCompletion.__annotations__["service_tier"] = NewServiceTierType
    ChatCompletion.model_fields["service_tier"] = FieldInfo(
        annotation=NewServiceTierType, default=None  # type: ignore
    )
    ChatCompletion.model_rebuild(force=True)


def patch_chat_cmpl_message_reasoning():
    ChatCompletionMessage.__annotations__["reasoning"] = typing.Optional[str]
    ChatCompletionMessage.model_fields["reasoning"] = FieldInfo(
        annotation=typing.Optional[str], default=None  # type: ignore
    )
    ChatCompletionMessage.model_rebuild(force=True)
