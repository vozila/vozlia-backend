# scripts/metrics_smoketest.py
"""
Usage:
  python scripts/metrics_smoketest.py /path/to/vozlia_metric_questions_500_db_based.md

This runs the metrics engine directly (no HTTP) and reports which questions are unsupported.

Notes:
- Requires DATABASE_URL and ADMIN_EMAIL (or existing users table) to be set, like your app.
- Intended for operator troubleshooting / regression testing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from db import SessionLocal
from services.user_service import get_or_create_primary_user
from services.metrics_service import run_metrics_question


def extract_questions(md: str) -> list[str]:
    out: list[str] = []
    for line in (md or "").splitlines():
        # match "1) ..." or "1. ..." etc.
        m = re.match(r"^\s*\d+\s*[\).\-\:]\s*(.+?)\s*$", line)
        if m:
            q = (m.group(1) or "").strip()
            if q:
                out.append(q)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Provide a markdown file path of metric questions.")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    md = path.read_text(encoding="utf-8", errors="ignore")
    questions = extract_questions(md)
    if not questions:
        print("No numbered questions found.")
        return 2

    db: Session = SessionLocal()
    try:
        user = get_or_create_primary_user(db)
        tenant_id = str(user.id)

        ok = 0
        bad: list[str] = []
        for q in questions:
            out = run_metrics_question(db, tenant_id=tenant_id, question=q, timezone="America/New_York")
            if out.get("ok"):
                ok += 1
            else:
                bad.append(q)

        print(f"Supported: {ok}/{len(questions)}")
        if bad:
            print("\nUnsupported questions:")
            for q in bad[:200]:
                print(f"- {q}")
            if len(bad) > 200:
                print(f"... plus {len(bad)-200} more")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
