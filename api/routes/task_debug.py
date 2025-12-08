from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from deps import get_db
from models import Task as TaskORM

router = APIRouter(tags=["task-debug"])


@router.get("/debug/tasks", response_model=List[TaskORM])
def list_all_tasks(db: Session = Depends(get_db)) -> List[TaskORM]:
    """
    Developer-only view of all tasks and their state.
    Gate this behind auth in production.
    """
    tasks = db.query(TaskORM).all()
    return tasks


@router.get("/debug/tasks/summary", response_model=Dict[str, Any])
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
