from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from deps import get_db
from models import Task as TaskORM
from tasks.interpreter import NLUEvent, handle_nlu_event  # ⬅️ NEW

router = APIRouter(tags=["task-debug"])


@router.get("/debug/tasks")
def list_all_tasks(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Developer-only view of all tasks and their state.
    NOTE: This returns JSON-encoded SQLAlchemy objects for debugging.
    """
    tasks = db.query(TaskORM).all()
    return jsonable_encoder(tasks)


@router.get("/debug/tasks/summary")
def tasks_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    tasks = db.query(TaskORM).all()
    by_status: Dict[str, int] = {}
    for t in tasks:
        key = t.status.value if hasattr(t.status, "value") else str(t.status)
        by_status[key] = by_status.get(key, 0) + 1

    return {
        "total": len(tasks),
        "by_status": by_status,
    }


@router.post("/debug/tasks/intent")
def debug_tasks_intent(
    event: NLUEvent,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Simulate 'brain' behavior:
      - take an intent + entities
      - run it through the task engine
      - return what Vozlia should say + task_id
    """
    directive = handle_nlu_event(db, event)
    return {
        "speech": directive.speech,
        "task_id": directive.task_id,
        "meta": directive.meta,
    }
