"""Собственный MCP-сервер с одним инструментом для учебного трекера."""

import sys
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tracker_api import app

mcp = FastMCP("demo-tracker")


@mcp.tool()
async def get_issue(issue_id: Annotated[str, Field(description="Ключ задачи, например DEMO-17",
                                                   pattern=r"^[A-Z]+-[0-9]+$")]) -> dict[str, str]:
    """Получить статус и исполнителя задачи по ключу или сообщение, что задача не найдена."""
    print(f"MCP → get_issue(issue_id={issue_id!r})", file=sys.stderr, flush=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://tracker.test") as client:
        response = await client.get(f"/issues/{issue_id}")
    print(f"API ← {response.request.method} {response.request.url.path} "
          f"→ HTTP {response.status_code} {response.text}", file=sys.stderr, flush=True)
    if response.status_code != 404:
        response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
