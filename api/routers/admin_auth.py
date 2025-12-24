# api/routers/admin_auth.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from core.jwt import create_jwt
from core.google import verify_google_id_token

router = APIRouter(prefix="/admin", tags=["admin"])

class AdminLogin(BaseModel):
    id_token: str

@router.post("/auth/login")
def admin_login(payload: AdminLogin):
    user = verify_google_id_token(payload.id_token)

    if user.email != os.getenv("ADMIN_EMAIL"):
        raise HTTPException(status_code=403, detail="Not authorized")

    token = create_jwt({
        "email": user.email,
        "role": "admin"
    })

    return {"token": token}
