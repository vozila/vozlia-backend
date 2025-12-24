# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.logging import logger

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Vozlia Backend")

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://admin.vozlia.com",
        "https://vozlia.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Required routers
# -----------------------
try:
    from api.routers.health import router as health_router
    app.include_router(health_router)
except Exception as e:
    logger.exception("FATAL: health router missing")
    raise

# -----------------------
# Optional routers
# -----------------------
try:
    from api.routers.me import router as me_router
except Exception:
    me_router = None
    logger.warning("Optional router missing: api.routers.me (skipping)")

try:
    from api.routers.email_accounts import router as email_accounts_router
except Exception:
    email_accounts_router = None
    logger.warning("Optional router missing: api.routers.email_accounts (skipping)")

try:
    from api.routers.oauth_google import router as oauth_google_router
except Exception:
    oauth_google_router = None
    logger.warning("Optional router missing: api.routers.oauth_google (skipping)")

try:
    from admin_google_oauth import router as admin_router
except Exception:
    admin_router = None
    logger.warning("Admin OAuth router missing (skipping)")

if me_router:
    app.include_router(me_router, prefix="/me")

if email_accounts_router:
    app.include_router(email_accounts_router, prefix="/me")

if oauth_google_router:
    app.include_router(oauth_google_router)

if admin_router:
    app.include_router(admin_router)

# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
async def startup():
    logger.info("Vozlia backend starting up")
