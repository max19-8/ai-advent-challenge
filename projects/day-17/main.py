"""Два независимых диалога с агентом и MCP-сервером трекера."""

import json
import os
import sys
from importlib import import_module
from pathlib import Path

import anyio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT))
day6 = import_module("projects.day-06.main")


def show(label, value):
    print(label, json.dumps(value, indent=2, ensure_ascii=False), sep="\n", flush=True)


class TrackerAgent(day6.Agent):
    system_prompt = "Отвечай по-русски. Сведения о задачах бери из трекера."

    def __init__(self, api_key, session, tool):
        super().__init__(api_key)
        self.session = session
        self.tool_name = tool.name
        self.tools = [{"type": "function", "function": {
            "name": tool.name, "description": tool.description, "parameters": tool.inputSchema,
        }}]

    def build_payload(self, messages):
        body = super().build_payload(messages)
        show("Запрос к модели", body)
        return body

    def request(self, messages):
        message, finish = super().request(messages)
        show(f"Ответ модели ({finish})", message)
        return message, finish

    def execute_tool(self, name, raw_arguments):
        if name != self.tool_name:
            return {"error": f"Неизвестный инструмент {name}"}
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Ожидался объект с аргументами")
        except ValueError as error:
            return {"error": str(error)}
        reply = anyio.from_thread.run(self.session.call_tool, name, arguments)
        if reply.isError:
            result = {"error": [part.model_dump(mode="json") for part in reply.content]}
        else:
            result = reply.structuredContent
        show("Результат MCP для модели", result)
        return result


async def run():
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Укажи POLZA_API_KEY в корневом .env")
    server = StdioServerParameters(command=sys.executable, args=[str(DIRECTORY / "server.py")])
    async with stdio_client(server) as streams, ClientSession(*streams) as session:
        info = await session.initialize()
        print(f"MCP-соединение установлено: {info.serverInfo.name}", flush=True)
        catalog = await session.list_tools()
        tool = next(t for t in catalog.tools if t.name == "get_issue")
        show("Зарегистрированный инструмент", tool.model_dump(mode="json", exclude_none=True))
        agent = TrackerAgent(api_key, session, tool)
        for issue_id in ("DEMO-17", "DEMO-404"):
            question = f"В каком статусе задача {issue_id} и кто за неё отвечает?"
            print("\nВопрос:", question, flush=True)
            answer = await anyio.to_thread.run_sync(agent.run, question)
            print("Финальный ответ:", answer, flush=True)


if __name__ == "__main__":
    anyio.run(run)
