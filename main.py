# main.py
from __future__ import annotations

import importlib
from fastapi import FastAPI

from core.logging import logger
from db import Base, engine

from api.routers.health import router as health_router
from api.routers.assistant import router as assistant_router
from api.routers.gmail_api import router as gmail_api_router
from api.routers.twilio import router as twilio_router

from vozlia_twilio.inbound import router as twilio_inbound_router
from vozlia_twilio.stream import twilio_stream

from admin_google_oauth import router as admin_router  # root-level admin router


def _maybe_include_router(app: FastAPI, module_path: str) -> None:
    """
    Optional router loader.
    - If module exists and has `router`, we include it.
    - If not, we log and continue (deploy never fails).
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logger.warning("Optional router missing: %s (skipping)", module_path)
        return

    router = getattr(mod, "router", None)
    if router is None:
        logger.warning("Module %s has no `router` attr (skipping)", module_path)
        return

    app.include_router(router)
    logger.info("Included optional router: %s", module_path)


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ensured (create_all).")

    # Required routers
    app.include_router(health_router)
    app.include_router(twilio_router)
    app.include_router(twilio_inbound_router)
    app.include_router(gmail_api_router)
    app.include_router(assistant_router)

    # Admin router (safe: admin endpoints themselves should gate on ADMIN_ENABLED)
    app.include_router(admin_router)

    # Twilio Media Streams WS
    app.add_api_websocket_route("/twilio/stream", twilio_stream)

    # Optional routers (won't break deploy if missing)
    _maybe_include_router(app, "api.routers.email_accounts")
    _maybe_include_router(app, "api.routers.oauth_google")

    return app


app = create_app()
