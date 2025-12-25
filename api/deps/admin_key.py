import os
from fastapi import Header, HTTPException

def require_admin_key(x_vozlia_admin_key: str | None = Header(default=None)):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected or x_vozlia_admin_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
