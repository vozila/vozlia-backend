from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from models import Task as TaskORM, TaskType, TaskStatus
from tasks.engine import (
    create_task as create_task_engine,
    update_task_inputs,
    execute_task as execute_task_engine,
    get_active_tasks_for_user,
)

import traceback

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskCreateRequest(BaseModel):
    user_id: str
    # accept plain string and convert to TaskType ourselves
    type: str
    inputs: Optional[Dict[str, Any]] = None


class TaskUpdateRequest(BaseModel):
    inputs: Dict[str, Any]


def _serialize_task(task: TaskORM) -> Dict[str, Any]:
    """
    Turn a SQLAlchemy Task into a JSON-serializable dict.
    """
    data = jsonable_encoder(task)
    # Make enum fields clean strings if needed
    if isinstance(task.type, TaskType):
        data["type"] = task.type.value
    if isinstance(task.status, TaskStatus):
        data["status"] = task.status.value
    return data


@router.post("/")
def create_task_endpoint(
    req: TaskCreateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    # Convert string → TaskType enum
    try:
        task_type = TaskType(req.type)
    except ValueError:
        valid = [t.value for t in TaskType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task type '{req.type}'. Valid values: {valid}",
        )

    try:
        task = create_task_engine(
            db,
            user_id=req.user_id,
            task_type=task_type,
            initial_inputs=req.inputs,
        )
    except Exception as e:
        tb = traceback.format_exc()
        # Surface the underlying error as JSON so we can see it in curl
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb,
            },
        )

    return {"task": _serialize_task(task)}


@router.get("/{task_id}")
def get_task_endpoint(
    task_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task": _serialize_task(task)}


@router.patch("/{task_id}")
def update_task_endpoint(
    task_id: str,
    req: TaskUpdateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    updated = update_task_inputs(db, task, req.inputs)
    return {"task": _serialize_task(updated)}


@router.post("/{task_id}/execute")
def execute_task_endpoint(
    task_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in (TaskStatus.READY, TaskStatus.EXECUTING, TaskStatus.WAITING):
        raise HTTPException(status_code=400, detail=f"Task is not ready: {task.status}")

    updated, summary = execute_task_engine(db, task)
    return {
        "task": _serialize_task(updated),
        "summary": summary,
    }


@router.get("/users/{user_id}/active")
def get_active_tasks_for_user_endpoint(
    user_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    tasks = get_active_tasks_for_user(db, user_id)
    return {"tasks": [_serialize_task(t) for t in tasks]}
