from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from db import SessionLocal
from admin_auth import require_admin

# Settings services (already in your repo)
from services.settings_service import (
    get_me_settings,
    update_agent_greeting,
    update_realtime_prompt_addendum,
    update_gmail_summary_enabled,
)

# Routers that already exist
from api.routers.health import router as health_router
from api.routers.user_settings import router as user_settings_router
from admin_google_oauth import router as admin_oauth_router

app = FastAPI()

# --- CORS (safe default; admin UI is backend-served) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(health_router)
app.include_router(user_settings_router)
app.include_router(admin_oauth_router)


# -------------------------------------------------------------------
# Admin Portal (server-rendered HTML)
# -------------------------------------------------------------------

@app.get("/admin", response_class=HTMLResponse)
def admin_portal(request: Request):
    """
    Simple backend-served admin portal.
    Auth required via existing admin OAuth.
    """
    require_admin(request)

    db = SessionLocal()
    settings = get_me_settings(db)
    db.close()

    return f"""
<!doctype html>
<html>
  <head>
    <title>Vozlia Admin</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        background: #f6f7f9;
        padding: 40px;
      }}
      .container {{
        max-width: 820px;
        margin: 0 auto;
        background: #fff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
      }}
      h1 {{
        margin-top: 0;
      }}
      label {{
        display: block;
        margin-top: 22px;
        font-weight: 600;
      }}
      textarea {{
        width: 100%;
        min-height: 120px;
        margin-top: 8px;
        padding: 12px;
        font-size: 14px;
        border-radius: 8px;
        border: 1px solid #ccc;
      }}
      .row {{
        margin-top: 18px;
      }}
      button {{
        margin-top: 26px;
        padding: 10px 18px;
        font-size: 14px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        background: #111;
        color: #fff;
      }}
      .hint {{
        font-size: 12px;
        color: #666;
        margin-top: 4px;
      }}
      .checkbox {{
        margin-top: 14px;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Vozlia Admin Portal</h1>

      <form method="post" action="/admin/save">

        <label>Agent Greeting</label>
        <textarea name="agent_greeting">{settings.agent_greeting or ""}</textarea>
        <div class="hint">Default greeting used by the assistant.</div>

        <label>Realtime Opening Rule (Addendum)</label>
        <textarea name="realtime_prompt_addendum">{settings.realtime_prompt_addendum or ""}</textarea>
        <div class="hint">
          Applied once at call start to steer the opening greeting (Flow A).
        </div>

        <div class="checkbox">
          <label>
            <input type="checkbox" name="gmail_summary_enabled"
              {"checked" if settings.gmail_summary_enabled else ""} />
            Enable Gmail summaries
          </label>
          <div class="hint">
            If disabled, the assistant will not summarize emails.
          </div>
        </div>

        <button type="submit">Save Settings</button>
      </form>
    </div>
  </body>
</html>
    """


@app.post("/admin/save")
def admin_save(
    request: Request,
    agent_greeting: str = Form(""),
    realtime_prompt_addendum: str = Form(""),
    gmail_summary_enabled: str | None = Form(None),
):
    """
    Persist admin settings and redirect back to portal.
    """
    require_admin(request)

    db = SessionLocal()

    update_agent_greeting(db, agent_greeting)
    update_realtime_prompt_addendum(db, realtime_prompt_addendum)
    update_gmail_summary_enabled(db, gmail_summary_enabled == "on")

    db.close()

    return RedirectResponse("/admin", status_code=303)
