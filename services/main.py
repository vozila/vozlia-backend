import logging
from fastapi import FastAPI

from db import Base, engine
from skills.loader import load_skills_from_disk

from api.routers.health import router as health_router
from api.routers.twilio import router as twilio_router, mount_twilio_ws
from api.routers.assistant import router as assistant_router
from api.routers.gmail_api import router as gmail_api_router
from api.routers.user_settings import router as user_settings_router

logger = logging.getLogger("vozlia")


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

