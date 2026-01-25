# main.py
from fastapi import FastAPI

from core.logging import logger  # IMPORTANT: initializes logging early

from db import Base, engine
from skills.loader import load_skills_from_disk

from api.routers.health import router as health_router
from api.routers.twilio import router as twilio_router, mount_twilio_ws
from api.routers.assistant import router as assistant_router
from api.routers.gmail_api import router as gmail_api_router
from api.routers.user_settings import router as user_settings_router
from api.routers.kb import router as kb_router
from api.routers.notify import router as notify_router
from api.routers.websearch import router as websearch_router
from api.routers.dbquery import router as dbquery_router
from api.routers.metrics import router as metrics_router


def create_app() -> FastAPI:
    app = FastAPI()

    # Core routes
    app.include_router(health_router)
    app.include_router(twilio_router)
    mount_twilio_ws(app)

    # Skill / helper routes
    app.include_router(assistant_router)
    app.include_router(gmail_api_router)

    # Settings routes (restores /me/settings/* endpoints used by the stream)
    app.include_router(user_settings_router)

    # KB router (placeholder health route + future KB endpoints)
    app.include_router(kb_router)

    # Notifications + WebSearch admin routes (control plane should proxy these)
    app.include_router(notify_router)
    app.include_router(websearch_router)
    app.include_router(dbquery_router)
    app.include_router(metrics_router)


    @app.on_event("startup")
    def _startup() -> None:
        # Ensure DB schema exists and load YAML skills.
        Base.metadata.create_all(bind=engine)
        load_skills_from_disk()
        logger.info("Database tables ensured and skills loaded.")

    @app.get("/")
    async def root():
        return {"ok": True}

    return app


app = create_app()
