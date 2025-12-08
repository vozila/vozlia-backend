from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List
from uuid import UUID  # ⬅️ NEW

from sqlalchemy.orm import Session

from models import Task as TaskORM, TaskStatus, TaskType
from .domain import TaskInputs, TaskState, TaskExecution


def _load_inputs(raw: Optional[Dict[str, Any]]) -> TaskInputs:
    return TaskInputs(**(raw or {}))


def _dump_inputs(inputs: TaskInputs) -> Dict[str, Any]:
    return inputs.model_dump()


def _load_state(raw: Optional[Dict[str, Any]]) -> TaskState:
    return TaskState(**(raw or {}))


def _dump_state(state: TaskState) -> Dict[str, Any]:
    return state.model_dump()


def _load_execution(raw: Optional[Dict[str, Any]]) -> TaskExecution:
    return TaskExecution(**(raw or {}))


def _dump_execution(exec_obj: TaskExecution) -> Dict[str, Any]:
    return exec_obj.model_dump()


def create_task(
    db: Session,
    *,
    user_id: str,
    task_type: TaskType,
    initial_inputs: Optional[Dict[str, Any]] = None,
) -> TaskORM:
    # Normalize user_id to a real UUID object for the DB
    try:
        user_uuid = UUID(str(user_id))
    except ValueError:
        raise ValueError(f"Invalid user_id '{user_id}', expected UUID string")

    # Task-type-specific config: required inputs, optional, steps
    if task_type == TaskType.REMINDER:
        required = ["time", "message"]
        optional = {}
        steps = ["confirm_details", "schedule"]
    elif task_type == TaskType.TIMER:
        required = ["duration"]
        optional = {}
        steps = ["confirm_duration", "start_timer"]
    elif task_type == TaskType.EMAIL_CHECK:
        required = ["query"]
        optional = {"max_results": 20}
        steps = ["retrieve", "summarize"]
    elif task_type == TaskType.NOTE:
        required = ["text"]
        optional = {}
        steps = ["save"]
    elif task_type == TaskType.COUNTING:
        required = ["start", "end"]
        optional = {}
        steps = ["count"]
    else:
        required = []
        optional = {}
        steps = []

    initial_inputs = initial_inputs or {}
    inputs = TaskInputs(
        required=required,
        optional=optional,
        collected={k: v for k, v in initial_inputs.items() if v is not None},
    )

    missing = inputs.missing_required()
    if missing:
        status = TaskStatus.COLLECTING_INPUT
        cursor = None
    else:
        status = TaskStatus.READY
        cursor = steps[0] if steps else None

    state = TaskState(cursor=cursor, context={"steps": steps}, history=[])
    execution = TaskExecution(result=None, error=None)

    task = TaskORM(
        user_id=user_uuid,  # ⬅️ use UUID, not raw string
        type=task_type,
        status=status,
        inputs=_dump_inputs(inputs),
        state=_dump_state(state),
        execution=_dump_execution(execution),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def update_task_inputs(db: Session, task: TaskORM, new_inputs: Dict[str, Any]) -> TaskORM:
    inputs = _load_inputs(task.inputs)
    state = _load_state(task.state)

    inputs.collected.update({k: v for k, v in new_inputs.items() if v is not None})
    missing = inputs.missing_required()

    if missing:
        task.status = TaskStatus.COLLECTING_INPUT
    else:
        task.status = TaskStatus.READY
        steps: List[str] = state.context.get("steps", [])
        if steps and state.cursor is None:
            state.cursor = steps[0]

    task.inputs = _dump_inputs(inputs)
    task.state = _dump_state(state)
    task.updated_at = datetime.utcnow()

    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_active_tasks_for_user(db: Session, user_id: str) -> list[TaskORM]:
    # Normalize to UUID for querying
    user_uuid = UUID(str(user_id))
    return (
        db.query(TaskORM)
        .filter(
            TaskORM.user_id == user_uuid,
            TaskORM.status.notin_([TaskStatus.COMPLETED, TaskStatus.ERROR]),
        )
        .all()
    )


def execute_task(db: Session, task: TaskORM) -> Tuple[TaskORM, str]:
    """
    Execute once and mark as COMPLETED for now.
    """
    if task.status not in (TaskStatus.READY, TaskStatus.EXECUTING, TaskStatus.WAITING):
        return task, "This task is not ready to execute yet."

    task.status = TaskStatus.EXECUTING
    inputs = _load_inputs(task.inputs)
    state = _load_state(task.state)
    execution = _load_execution(task.execution)

    try:
        if task.type == TaskType.REMINDER:
            summary = _execute_reminder(inputs)
        elif task.type == TaskType.TIMER:
            summary = _execute_timer(inputs)
        elif task.type == TaskType.EMAIL_CHECK:
            summary = _execute_email_check(inputs)
        elif task.type == TaskType.NOTE:
            summary = _execute_note(inputs)
        elif task.type == TaskType.COUNTING:
            summary = _execute_counting(inputs, state)
        else:
            summary = f"I’ve recorded a task of type {task.type}."

        execution.result = summary
        execution.error = None
        task.status = TaskStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        execution.error = str(exc)
        task.status = TaskStatus.ERROR
        summary = "Something went wrong while running that task."

    task.execution = _dump_execution(execution)
    task.state = _dump_state(state)
    task.updated_at = datetime.utcnow()

    db.add(task)
    db.commit()
    db.refresh(task)
    return task, summary


# === Per-type behaviors ===

def _execute_reminder(inputs: TaskInputs) -> str:
    time_val = inputs.collected.get("time")
    message = inputs.collected.get("message", "")
    # TODO: real scheduler
    return f"Okay, I’ll remind you at {time_val}: {message}"


def _execute_timer(inputs: TaskInputs) -> str:
    duration = inputs.collected.get("duration")
    # TODO: real timer
    return f"Alright, I’ve started a timer for {duration}."


def _execute_email_check(inputs: TaskInputs) -> str:
    query = inputs.collected.get("query")
    max_results = inputs.collected.get("max_results", 20)
    # TODO: hook into your email module later
    return f"I’ll check your email for '{query}' and summarize the top {max_results} results."


def _execute_note(inputs: TaskInputs) -> str:
    text = inputs.collected.get("text")
    # TODO: store note somewhere if you like
    return f"I’ve saved your note: {text}"


def _execute_counting(inputs: TaskInputs, state: TaskState) -> str:
    start = int(inputs.collected.get("start"))
    end = int(inputs.collected.get("end"))
    current = int(state.context.get("current", start))
    state.context["current"] = current
    return f"I’m counting from {start} to {end}. I’m currently at {current}."
