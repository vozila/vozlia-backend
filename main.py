# main.py
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.logging import logger
from db import Base, engine

from api.routers.health import router as health_router
from api.routers.twilio import router as twilio_router
try:
    from api.routers.oauth_google import router as oauth_google_router
except ModuleNotFoundError:
    oauth_google_router = None
from api.routers.email_accounts import router as email_accounts_router
from api.routers.gmail_api import router as gmail_router
from api.routers.assistant import router as assistant_router


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ensured (create_all).")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        try:
            body = await request.body()
        except Exception:
            body = b"<unreadable>"
        logger.error(
            "422 VALIDATION ERROR path=%s errors=%s body=%r",
            request.url.path,
            exc.errors(),
            body[:2000],
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    # Routers
    app.include_router(health_router)
    app.include_router(twilio_router)
    app.include_router(oauth_google_router)
    app.include_router(email_accounts_router)
    app.include_router(gmail_router)
    app.include_router(assistant_router)
    from api.routers.twilio import mount_twilio_ws
    ...
    app.include_router(twilio_router)
    mount_twilio_ws(app)

    return app


app = create_app()
