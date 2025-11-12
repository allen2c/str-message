import typing

from agents import HostedMCPTool
from openai.types.responses.tool import Mcp
from openai.types.responses.tool_param import Mcp as McpParam

aws_knowledge_mcp = Mcp(
    server_label="aws-knowledge-mcp-server",
    type="mcp",
    server_description="AWS knowledge base for software development tools and resources.",  # noqa: E501
    server_url="https://knowledge-mcp.global.api.aws",
)
aws_knowledge_mcp_param, aws_knowledge_mcp_tool = (
    McpParam(**aws_knowledge_mcp.model_dump()),
    HostedMCPTool(
        tool_config=McpParam(**aws_knowledge_mcp.model_dump()),
    ),
)


mcp_map: typing.Dict[str, Mcp] = {
    aws_knowledge_mcp.server_label: aws_knowledge_mcp,
}
mcp_param_map: typing.Dict[str, McpParam] = {
    aws_knowledge_mcp.server_label: aws_knowledge_mcp_param,
}
mcp_tool_map: typing.Dict[str, HostedMCPTool] = {
    aws_knowledge_mcp.server_label: aws_knowledge_mcp_tool,
}
