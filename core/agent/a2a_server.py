from __future__ import annotations

import os

import ollama
import uvicorn
from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import RequestHandler
from fastapi import FastAPI

from core.agent.card import get_agent_card
from core.agent.handler import MediaAgentHandler
from core.utils.logger import configure_logger, get_logger

configure_logger()
logger = get_logger(__name__)


def check_ollama_connection(model_name: str) -> None:
    try:
        ollama.list()
        logger.info("ollama.ok", model=model_name)
    except Exception as exc:
        logger.error("ollama.failed", error=str(exc))
        raise RuntimeError("Ollama is not running") from exc


def create_app() -> FastAPI:
    base_url = os.getenv("MEDIA_AGENT_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MEDIA_AGENT_MODEL", "llama3.1")

    check_ollama_connection(model_name)

    agent_card = get_agent_card(base_url=base_url)
    handler: RequestHandler = MediaAgentHandler(model_name=model_name)

    app_builder = A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    app = app_builder.build(rpc_url="/a2a")

    logger.info("a2a.server.started", base_url=base_url, model=model_name)

    return app


app = create_app()


@app.middleware("http")
async def log_routes(request, call_next):
    if not getattr(app.state, "_routes_logged", False):
        app.state._routes_logged = True
        for route in app.routes:
            path = getattr(route, "path", None)
            if path:
                logger.info("route", path=path)
    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run(
        "core.agent.a2a_server:app",
        host="0.0.0.0",
        port=int(os.getenv("MEDIA_AGENT_PORT", "8000")),
        # reload=True,
        factory=False,
    )
