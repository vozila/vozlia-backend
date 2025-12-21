# api/routers/health.py
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()

@router.get("/")
async def root():
    return PlainTextResponse("OK")

@router.get("/health")
async def health():
    return {"status": "ok"}
