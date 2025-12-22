# skills/models.py
from __future__ import annotations

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class SkillTrigger(BaseModel):
    phrases: List[str] = Field(default_factory=list)


class SkillAPI(BaseModel):
    type: str  # "internal" | "http"
    handler: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = "GET"


class SkillPrompt(BaseModel):
    system: Optional[str] = None
    user: Optional[str] = None


class SkillResponse(BaseModel):
    speak: Optional[str] = None


class SkillDefinition(BaseModel):
    id: str
    name: str
    greeting: Optional[str] = None
    trigger: SkillTrigger
    api: SkillAPI
    prompt: Optional[SkillPrompt] = None
    response: Optional[SkillResponse] = None
