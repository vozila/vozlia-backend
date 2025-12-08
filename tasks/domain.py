from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from models import TaskStatus, TaskType


class TaskInputs(BaseModel):
    required: List[str] = Field(default_factory=list)
    optional: Dict[str, Any] = Field(default_factory=dict)
    collected: Dict[str, Any] = Field(default_factory=dict)

    def missing_required(self) -> List[str]:
        return [k for k in self.required if k not in self.collected]


class TaskState(BaseModel):
    cursor: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class TaskExecution(BaseModel):
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskRead(BaseModel):
    """
    Optional Pydantic response model if you want more control over API responses.
    Not strictly required yet, but ready if you want to use it.
    """
    id: str
    user_id: str
    type: TaskType
    status: TaskStatus
    inputs: TaskInputs
    state: TaskState
    execution: TaskExecution
    created_at: datetime
    updated_at: datetime

    class Config:
        arbitrary_types_allowed = True
