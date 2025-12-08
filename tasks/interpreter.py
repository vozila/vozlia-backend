from __future__ import annotations

from typing import Any, Dict, Optional, List

from pydantic import BaseModel
from sqlalchemy.orm import Session

from models import TaskType, TaskStatus, Task as TaskORM
from .engine import (
    create_task,
    update_task_inputs,
    execute_task,
    get_active_tasks_for_user,
)


class NLUEvent(BaseModel):
    """
    Represents an intent extracted from caller speech.
    You’ll construct this from Realtime events later.
    """
    event: str  # e.g. "task.intent"
    intent: str
    text: Optional[str] = None
    user_id: str
    task_id: Optional[str] = None
    entities: Dict[str, Any] = {}


class AssistantDirective(BaseModel):
    """
    What your backend tells the assistant to say/do.
    """
    speech: str
    task_id: Optional[str] = None
    meta: Dict[str, Any] = {}


INTENT_TO_TASK_TYPE = {
    "set_reminder": TaskType.REMINDER,
    "start_timer": TaskType.TIMER,
    "take_note": TaskType.NOTE,
    "check_email": TaskType.EMAIL_CHECK,
    "start_counting": TaskType.COUNTING,
    "start_onboarding": TaskType.WORKFLOW,
}


def handle_nlu_event(db: Session, evt: NLUEvent) -> AssistantDirective:
    """
    Main entrypoint: given intent + entities + (optional) task_id,
    decide what to do with the task engine and return a directive.
    """
    if evt.intent == "continue_task":
        return _handle_continue(db, evt)
    if evt.intent == "cancel_task":
        return _handle_cancel(db, evt)
    if evt.intent == "what_number_were_you_on":
        return _handle_query_current(db, evt)

    task_type = INTENT_TO_TASK_TYPE.get(evt.intent)
    if not task_type:
        return AssistantDirective(
            speech="I understood what you said, but I’m not sure how to turn that into a task yet.",
            meta={"intent": evt.intent},
        )

    # If an existing task is referenced -> update that
    if evt.task_id:
        task = db.query(TaskORM).filter(TaskORM.id == evt.task_id).first()
        if task:
            updated_task = update_task_inputs(db, task, evt.entities or {})
            missing = _missing_required(updated_task)
            if missing:
                return AssistantDirective(
                    speech=_prompt_for_missing(missing),
                    task_id=str(updated_task.id),
                    meta={"missing": missing},
                )

            updated_task, summary = execute_task(db, updated_task)
            return AssistantDirective(
                speech=summary,
                task_id=str(updated_task.id),
                meta={"status": updated_task.status},
            )

    # Otherwise, create new task
    task = create_task(
        db,
        user_id=evt.user_id,
        task_type=task_type,
        initial_inputs=evt.entities,
    )

    missing = _missing_required(task)
    if missing:
        return AssistantDirective(
            speech=_prompt_for_missing(missing),
            task_id=str(task.id),
            meta={"missing": missing},
        )

    task, summary = execute_task(db, task)
    return AssistantDirective(
        speech=summary,
        task_id=str(task.id),
        meta={"status": task.status},
    )


def _missing_required(task: TaskORM) -> List[str]:
    inputs = TaskInputsWrapper.from_raw(task.inputs)
    return inputs.missing_required()


def _prompt_for_missing(missing: List[str]) -> str:
    if len(missing) == 1:
        field = missing[0]
        if field == "time":
            return "Sure. What time should I set it for?"
        if field == "message":
            return "Okay. What should the reminder say?"
        if field == "duration":
            return "How long should I set the timer for?"
        if field == "text":
            return "What would you like me to write down?"
        if field == "query":
            return "What should I look for in your email?"
        if field == "start":
            return "What number should I start counting from?"
        if field == "end":
            return "What number should I stop at?"
    return "I need a bit more information. Can you tell me the details I’m missing?"


def _get_current_task_for_user(db: Session, user_id: str, task_id: Optional[str]) -> Optional[TaskORM]:
    if task_id:
        task = db.query(TaskORM).filter(TaskORM.id == task_id).first()
        if task:
            return task

    active = get_active_tasks_for_user(db, user_id)
    return active[0] if active else None


def _handle_continue(db: Session, evt: NLUEvent) -> AssistantDirective:
    task = _get_current_task_for_user(db, evt.user_id, evt.task_id)
    if not task:
        return AssistantDirective(speech="There’s nothing in progress to continue right now.")

    if task.type == TaskType.COUNTING:
        inputs = TaskInputsWrapper.from_raw(task.inputs)
        state = TaskStateWrapper.from_raw(task.state)

        start = int(inputs.collected.get("start"))
        end = int(inputs.collected.get("end"))
        current = int(state.context.get("current", start))

        if current >= end:
            task.status = TaskStatus.COMPLETED
            db.add(task)
            db.commit()
            return AssistantDirective(
                speech="I’ve already finished counting.",
                task_id=str(task.id),
                meta={"status": task.status},
            )

        current += 1
        state.context["current"] = current
        task.state = state.to_raw()
        db.add(task)
        db.commit()

        return AssistantDirective(
            speech=f"{current}",
            task_id=str(task.id),
            meta={"status": task.status, "current": current},
        )

    # For other workflows, we can later add step-based logic
    return AssistantDirective(
        speech="Okay, let’s pick up where we left off.",
        task_id=str(task.id),
        meta={"status": task.status},
    )


def _handle_cancel(db: Session, evt: NLUEvent) -> AssistantDirective:
    task = _get_current_task_for_user(db, evt.user_id, evt.task_id)
    if not task:
        return AssistantDirective(speech="There’s no active task to cancel.")

    task.status = TaskStatus.ERROR
    exec_obj = (task.execution or {}) if isinstance(task.execution, dict) else {}
    exec_obj["error"] = "Cancelled by user"
    task.execution = exec_obj

    db.add(task)
    db.commit()

    return AssistantDirective(
        speech="Okay, I’ve cancelled that.",
        task_id=str(task.id),
        meta={"status": task.status},
    )


def _handle_query_current(db: Session, evt: NLUEvent) -> AssistantDirective:
    active = get_active_tasks_for_user(db, evt.user_id)
    counting_tasks = [t for t in active if t.type == TaskType.COUNTING]
    task = counting_tasks[0] if counting_tasks else None

    if not task:
        return AssistantDirective(speech="I’m not currently counting anything.")

    state = TaskStateWrapper.from_raw(task.state)
    inputs = TaskInputsWrapper.from_raw(task.inputs)
    start = int(inputs.collected.get("start"))
    current = int(state.context.get("current", start))

    return AssistantDirective(
        speech=f"I was on {current}.",
        task_id=str(task.id),
        meta={"current": current},
    )


# --- Tiny wrapper helpers to re-use domain logic with raw JSON blobs ---

class TaskInputsWrapper(TaskInputs):
    @classmethod
    def from_raw(cls, raw: Dict[str, Any] | None) -> "TaskInputsWrapper":
        return cls(**(raw or {}))


class TaskStateWrapper(TaskState):
    @classmethod
    def from_raw(cls, raw: Dict[str, Any] | None) -> "TaskStateWrapper":
        return cls(**(raw or {}))

    def to_raw(self) -> Dict[str, Any]:
        return self.model_dump()
