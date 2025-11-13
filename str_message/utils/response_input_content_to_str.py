import logging
import typing

from openai.types.responses.response_function_call_output_item import (
    ResponseFunctionCallOutputItem,
)
from openai.types.responses.response_function_call_output_item_list import (
    ResponseFunctionCallOutputItemList,
)
from openai.types.responses.response_input_content import ResponseInputContent
from openai.types.responses.response_input_file import ResponseInputFile
from openai.types.responses.response_input_file_content import ResponseInputFileContent
from openai.types.responses.response_input_image import ResponseInputImage
from openai.types.responses.response_input_image_content import (
    ResponseInputImageContent,
)
from openai.types.responses.response_input_message_content_list import (
    ResponseInputMessageContentList,
)
from openai.types.responses.response_input_text import ResponseInputText
from openai.types.responses.response_input_text_content import ResponseInputTextContent
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText

logger = logging.getLogger(__name__)


def response_input_content_to_str(
    content: typing.Union[
        str,
        ResponseInputMessageContentList,
        ResponseInputContent,
        ResponseOutputText,
        ResponseOutputRefusal,
        typing.List[typing.Union[ResponseOutputText, ResponseOutputRefusal]],
        ResponseFunctionCallOutputItemList,
        ResponseFunctionCallOutputItem,
        ResponseInputFileContent,
        ResponseInputTextContent,
        ResponseInputImageContent,
    ],
) -> str:
    from str_message import (
        CONTENT_FILE_FILENAME_EXPR,
        CONTENT_FILE_ID_EXPR,
        CONTENT_FILE_URL_EXPR,
        CONTENT_IMAGE_ID_EXPR,
        CONTENT_IMAGE_URL_EXPR,
    )

    if isinstance(content, str):
        return content

    elif isinstance(content, typing.List):
        return "\n\n".join(
            response_input_content_to_str(response_input_content)
            for response_input_content in content
        )

    elif isinstance(content, ResponseInputText):
        return content.text

    elif isinstance(content, ResponseInputImage):
        if content.file_id is not None:
            return CONTENT_IMAGE_ID_EXPR.format(image_id=content.file_id)
        elif content.image_url is not None:
            return CONTENT_IMAGE_URL_EXPR.format(image_url=content.image_url)
        else:
            raise ValueError(f"Unsupported image content: {content}")

    elif isinstance(content, ResponseInputFile):
        if content.file_id is not None:
            return CONTENT_FILE_ID_EXPR.format(file_id=content.file_id)
        elif content.file_url is not None:
            return CONTENT_FILE_URL_EXPR.format(file_url=content.file_url)
        elif content.file_data is not None:
            return CONTENT_FILE_URL_EXPR.format(file_url=content.file_data)
        elif content.filename is not None:
            return CONTENT_FILE_FILENAME_EXPR.format(filename=content.filename)
        else:
            raise ValueError(f"Unsupported file content: {content}")

    elif isinstance(content, ResponseOutputText):
        return content.text

    elif isinstance(content, ResponseOutputRefusal):
        return content.refusal

    elif isinstance(content, ResponseInputFileContent):
        if content.file_id is not None:
            return CONTENT_FILE_ID_EXPR.format(file_id=content.file_id)
        elif content.file_url is not None:
            return CONTENT_FILE_URL_EXPR.format(file_url=content.file_url)
        elif content.file_data is not None:
            return CONTENT_FILE_URL_EXPR.format(file_url=content.file_data)
        elif content.filename is not None:
            return CONTENT_FILE_FILENAME_EXPR.format(filename=content.filename)
        else:
            raise ValueError(f"Unsupported file content: {content}")

    elif isinstance(content, ResponseInputTextContent):
        return content.text

    elif isinstance(content, ResponseInputImageContent):
        if content.file_id is not None:
            return CONTENT_IMAGE_ID_EXPR.format(image_id=content.file_id)
        elif content.image_url is not None:
            return CONTENT_IMAGE_URL_EXPR.format(image_url=content.image_url)
        else:
            raise ValueError(f"Unsupported image content: {content}")

    else:
        raise ValueError(f"Unsupported content type: {type(content)}")
