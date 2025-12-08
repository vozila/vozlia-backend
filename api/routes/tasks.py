from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException
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

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskCreateRequest(BaseModel):
    user_id: str
    type: TaskType
    inputs: Optional[Dict[str, Any]] = None


class TaskUpdateRequest(BaseModel):
    inputs: Dict[str, Any]


@router.post("/", response_model=Dict[str, Any])
def create_task_endpoint(req: TaskCreateRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    task = create_task_engine(
        db,
        user_id=req.user_id,
        task_type=req.type,
        initial_inputs=req.inputs,
    )
    return {"task": task}


@router.get("/{task_id}", response_model=Dict[str, Any])
def get_task_endpoint(task_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task": task}


@router.patch("/{task_id}", response_model=Dict[str, Any])
def update_task_endpoint(task_id: str, req: TaskUpdateRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    updated = update_task_inputs(db, task, req.inputs)
    return {"task": updated}


@router.post("/{task_id}/execute", response_model=Dict[str, Any])
def execute_task_endpoint(task_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in (TaskStatus.READY, TaskStatus.EXECUTING, TaskStatus.WAITING):
        raise HTTPException(status_code=400, detail=f"Task is not ready: {task.status}")

    updated, summary = execute_task_engine(db, task)
    return {"task": updated, "summary": summary}


@router.get("/users/{user_id}/active", response_model=Dict[str, Any])
def get_active_tasks_for_user_endpoint(user_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    tasks = get_active_tasks_for_user(db, user_id)
    return {"tasks": tasks}
