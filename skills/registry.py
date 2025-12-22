# skills/registry.py
from __future__ import annotations

from typing import Dict, List
from skills.models import SkillDefinition


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.id] = skill

    def all(self) -> List[SkillDefinition]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> SkillDefinition | None:
        return self._skills.get(skill_id)


# Singleton registry (safe for now)
skill_registry = SkillRegistry()
