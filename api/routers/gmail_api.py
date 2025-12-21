# api/routers/gmail_api.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from deps import get_db
from services.user_service import get_or_create_demo_user
from services.gmail_service import gmail_list_messages, summarize_gmail_for_assistant, get_gmail_account_or_404, ensure_gmail_access_token
from core import config as cfg
import httpx

router = APIRouter(prefix="/email")

@router.get("/accounts/{account_id}/messages")
def list_gmail_messages(account_id: str, max_results: int = 20, query: str | None = None, db: Session = Depends(get_db)):
    user = get_or_create_demo_user(db)
    return gmail_list_messages(account_id, user, db, max_results=max_results, query=query)

@router.get("/accounts/{account_id}/summary")
def gmail_summary(account_id: str, max_results: int = 20, query: str | None = None, db: Session = Depends(get_db)):
    user = get_or_create_demo_user(db)
    data = summarize_gmail_for_assistant(account_id, user, db, max_results=max_results, query=query)
    data["messages"] = data.get("messages", [])[:max_results]
    return data

@router.get("/accounts/{account_id}/stats")
def gmail_stats(account_id: str, window_days: int = 1, db: Session = Depends(get_db)):
    if window_days <= 0:
        return {"detail": "window_days must be >= 1"}

    user = get_or_create_demo_user(db)
    account = get_gmail_account_or_404(account_id, user, db)
    access_token = ensure_gmail_access_token(account, db)

    query = f"newer_than:{window_days}d"
    with httpx.Client(timeout=10.0) as client_http:
        list_url = f"{cfg.GMAIL_API_BASE}/users/me/messages"
        list_resp = client_http.get(
            list_url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"q": query, "maxResults": 1},
        )
        list_resp.raise_for_status()
        data = list_resp.json()
        size_estimate = data.get("resultSizeEstimate", 0)

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "window_days": window_days,
        "query": query,
        "approx_message_count": size_estimate,
    }
