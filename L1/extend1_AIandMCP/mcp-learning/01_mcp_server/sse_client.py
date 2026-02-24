import sys
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client
from typing import Any
import logging
import mcp.types as types
from pydantic import AnyUrl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPClient:
    def __init__(self):
        self._session_context = None
        self._streams_context = None
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_sse_server(self, server_url: str):
        """通过 sse 传输方式连接到 MCP 服务端"""
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # 初始化
        await self.session.initialize()

    async def list_tools(self):
        """列出全部工具"""
        try:
            response = await self.session.list_tools()
            tools = response.tools
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any]
    ) -> Any:
        """调用工具"""
        try:
            result = await self.session.call_tool(tool_name, arguments)
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return result

    async def list_prompts(self):
        """列出全部提示模板"""
        try:
            prompt_list = await self.session.list_prompts()
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return prompt_list

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None):
        """读取提示模板内容"""
        try:
            prompt = await self.session.get_prompt(name=name, arguments=arguments)
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return prompt

    async def list_resources(self) -> types.ListResourcesResult:
        """列出全部资源"""
        try:
            list_resources = await self.session.list_resources()
        except Exception as e:
            error_msg = f"Error list resources: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return list_resources

    async def list_resource_templates(self) -> types.ListResourceTemplatesResult:
        """列出全部带参数的资源"""
        try:
            list_resource_templates = await self.session.list_resource_templates()
        except Exception as e:
            error_msg = f"Error list resource templates: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return list_resource_templates

    async def read_resource(self, uri: AnyUrl) -> types.ReadResourceResult:
        """读取资源"""
        try:
            resource_datas = await self.session.read_resource(uri=uri)
        except Exception as e:
            error_msg = f"Error list resource templates: {str(e)}"
            logging.error(error_msg)
            return error_msg
        return resource_datas

    async def cleanup(self):
        """关闭会话和连接流"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python sse_client.py <URL of SSE MCP server (i.e. http://localhost:8002/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        # 列出全部 tools
        tools = await client.list_tools()
        print('------------列出全部 tools')
        for tool in tools:
            print(f'---- 工具名称：{tool.name}, 描述：{tool.description}')
            print(f"输入参数: {tool.inputSchema}")

        # 调用工具
        result = await client.execute_tool('add', {'a': 2, 'b': 3})
        print(f'工具执行结果：{result}')

        # 调用全部 prompts
        prompts_list = await client.list_prompts()
        print('------------列出全部 prompts')
        for prompt in prompts_list.prompts:
            print(f'---- prompt 名称: {prompt.name}, 描述：{prompt.description}, 参数：{prompt.arguments}')

        # 获取 "介绍中国省份" prompt 内容
        province_name = '四川省'
        prompt_result = await client.get_prompt(name='introduce_china_province', arguments={'province':province_name})
        prompt_content = prompt_result.messages[0].content.text
        print(f'-------介绍{province_name}的 prompt：{prompt_content}')

        # 列出全部的 resources
        resources_list = await client.list_resources()
        print('---- 列出全部 resources')
        print(resources_list.resources)

        # 列出全部的 resource templates
        resource_templates_list = await client.list_resource_templates()
        print('---- 列出全部 resource templates')
        print(resource_templates_list.resourceTemplates)

        # 获取全部数据表的表名
        uri = AnyUrl('db://tables')
        table_names = await client.read_resource(uri)
        print('---- 全部数据表：')
        print(table_names.contents[0].text)

        # 读取某个数据表的数据
        uri = AnyUrl("db://tables/chinese_movie_ratings/data/10")
        resource_datas = await client.read_resource(uri)
        print('chinese_movie_ratings 表数据：')
        print(resource_datas.contents[0].text)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
