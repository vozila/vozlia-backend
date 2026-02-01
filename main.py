"""VOZLIA FILE PURPOSE
Purpose: FastAPI app bootstrap + route registration only.
Hot path: no
Public interfaces: create_app(), app
Reads/Writes: none directly (routers/services handle IO).
Feature flags: n/a (routers may gate internally).
Failure mode: startup errors prevent serving.
Last touched: 2026-02-01 (add admin concepts router behind CONCEPTS_ENABLED)
"""

# main.py
from fastapi import FastAPI

from core.logging import logger  # IMPORTANT: initializes logging early

from db import Base, engine, SessionLocal
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
from api.routers.concepts import router as concepts_router

# Admin troubleshooting routes
from api.routers.admin_settings import router as admin_settings_router
from api.routers.admin_dynamic_skills import router as admin_dynamic_skills_router


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
    app.include_router(concepts_router)

    # Admin settings + troubleshooting
    app.include_router(admin_settings_router)
    app.include_router(admin_dynamic_skills_router)

    @app.on_event("startup")
    def _startup() -> None:
        # Ensure DB schema exists and load YAML skills.
        Base.metadata.create_all(bind=engine)
        load_skills_from_disk()

        # Best-effort: repair dynamic-skill routing after rollbacks / drift.
        # Rollback: set DYNAMIC_SKILLS_AUTOSYNC=0 to disable.
        try:
            from services.dynamic_skill_autosync import dynamic_skills_autosync_enabled, autosync_dynamic_skills
            from services.user_service import get_or_create_primary_user

            if dynamic_skills_autosync_enabled():
                db = SessionLocal()
                try:
                    user = get_or_create_primary_user(db)
                    autosync_dynamic_skills(db, user, force=False)
                    db.commit()
                finally:
                    db.close()
        except Exception:
            # Never crash startup due to autosync; it is purely a recovery mechanism.
            logger.exception("DYNAMIC_SKILLS_AUTOSYNC_STARTUP_FAIL")

        logger.info("Database tables ensured and skills loaded.")

    @app.get("/")
    async def root():
        return {"ok": True}

    return app


app = create_app()
