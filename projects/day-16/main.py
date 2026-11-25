"""Агент Дня 6 с инструментом официального Filesystem MCP Server."""

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
DATA = DIRECTORY / "data"
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT))
day6 = import_module("projects.day-06.main")

READ = "read_text_file"
CALCULATOR = next(tool for tool in day6.TOOLS if tool["function"]["name"] == "calculate")


def show(label, value):
    print(label, json.dumps(value, ensure_ascii=False, indent=2), sep="\n", flush=True)


class FilesystemAgent(day6.Agent):
    system_prompt = (
        "Отвечай по-русски. Учебный курс возьми из указанного файла. "
        "Сам выбирай доступные инструменты и аргументы. "
        "Если данных получить не удалось, сообщи об этом."
    )

    def __init__(self, api_key, session, reader, mixed):
        super().__init__(api_key)
        self.session = session
        self.mixed = mixed
        self.tools = [{"type": "function", "function": {
            "name": reader.name,
            "description": reader.description,
            "parameters": reader.inputSchema,
        }}]
        if mixed:
            self.tools.append(CALCULATOR)

    def build_payload(self, messages):
        body = super().build_payload(messages)
        show("Запрос к модели", body)
        return body

    def request(self, messages):
        message, finish = super().request(messages)
        show(f"Ответ модели ({finish})", message)
        return message, finish

    def execute_tool(self, name, raw_arguments):
        if name == READ:
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Ожидался объект с аргументами")
                # Цикл Дня 6 работает в worker-потоке, MCP-сессия — в цикле AnyIO.
                reply = anyio.from_thread.run(self.session.call_tool, name, arguments)
                result = {"isError": reply.isError, "content": [
                    item.model_dump(mode="json", exclude_none=True) for item in reply.content
                ]}
            except ValueError as error:
                result = {"error": str(error)}
            source = "MCP"
        elif self.mixed and name == "calculate":
            result = super().execute_tool(name, raw_arguments)
            source = "локальный Python"
        else:
            result = {"error": f"Инструмент {name} недоступен в этом сценарии"}
            source = "агент"
        show(f"{source} → {name}", result)
        return result


async def run():
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("POLZA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Укажи POLZA_API_KEY в корневом .env")
    server = StdioServerParameters(
        command="npx", args=["--yes", "@modelcontextprotocol/server-filesystem", str(DATA)],
        cwd=DATA,
    )
    async with stdio_client(server) as (stream_in, stream_out):
        async with ClientSession(stream_in, stream_out) as session:
            info = await session.initialize()
            show("MCP-соединение установлено", info.serverInfo.model_dump(mode="json"))
            catalog = await session.list_tools()
            tools = list(catalog.tools)
            while catalog.nextCursor:
                catalog = await session.list_tools(cursor=catalog.nextCursor)
                tools.extend(catalog.tools)
            show("Инструменты MCP-сервера", [tool.name for tool in tools])
            reader = next((tool for tool in tools if tool.name == READ), None)
            if reader is None:
                raise RuntimeError(f"В каталоге MCP отсутствует {READ}")

            scenarios = [
                (False, "Какой учебный курс USD к RUB указан в файле rates.txt?"),
                (True, "Сколько рублей стоят 237 долларов по курсу из rates.txt? "
                       "Для расчёта воспользуйся калькулятором."),
            ]
            for mixed, question in scenarios:
                print("\nСценарий:", "MCP + calculate" if mixed else "только MCP", flush=True)
                agent = FilesystemAgent(api_key, session, reader, mixed)
                answer = await anyio.to_thread.run_sync(agent.run, question)
                print("Финальный ответ:", answer, flush=True)


if __name__ == "__main__":
    anyio.run(run)
