# skills/loader.py
from __future__ import annotations

import yaml
from pathlib import Path
from skills.models import SkillDefinition
from skills.registry import skill_registry
from core.logging import logger


SKILLS_DIR = Path(__file__).parent / "manifests"


def load_skills_from_disk() -> None:
    if not SKILLS_DIR.exists():
        logger.warning("Skills directory missing: %s", SKILLS_DIR)
        return

    for path in SKILLS_DIR.glob("*.yaml"):
        try:
            with path.open("r") as f:
                data = yaml.safe_load(f)

            skill = SkillDefinition.model_validate(data)
            skill_registry.register(skill)

            logger.info("Loaded skill: %s (%s)", skill.id, path.name)

        except Exception as e:
            logger.exception("Failed to load skill %s: %s", path.name, e)
