"""Учебный HTTP API трекера. Данные задачи принадлежат только API."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


ISSUES = {
    "DEMO-17": {"id": "DEMO-17", "status": "На проверке", "assignee": "Анна Соколова"},
}


async def issue(request: Request):
    issue_id = request.path_params["issue_id"]
    if issue_id not in ISSUES:
        return JSONResponse({"id": issue_id, "error": "Задача не найдена"}, status_code=404)
    return JSONResponse(ISSUES[issue_id])


app = Starlette(routes=[Route("/issues/{issue_id}", issue, methods=["GET"])])
