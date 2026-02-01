"""VOZLIA FILE PURPOSE
Purpose: Tenant-scoped storage helpers for Concept Codes (definitions, assignments, batches).
Hot path: no
Public interfaces: concepts_enabled, upsert/list functions.
Reads/Writes: concept_definitions, concept_assignments, concept_batches
Feature flags: CONCEPTS_ENABLED
Failure mode: raises ValueError for invalid inputs; callers should translate to HTTP errors.
Last touched: 2026-02-01 (initial Concept Code store)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.logging import logger
from models import User, ConceptDefinition, ConceptAssignment, ConceptBatch


def concepts_enabled() -> bool:
    v = (os.getenv("CONCEPTS_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _now_utc() -> datetime:
    return datetime.utcnow()


def list_concept_definitions(db: Session, user: User, *, active_only: bool = False) -> list[ConceptDefinition]:
    q = db.query(ConceptDefinition).filter(ConceptDefinition.tenant_id == user.id)
    if active_only:
        q = q.filter(ConceptDefinition.active == True)  # noqa: E712
    return q.order_by(ConceptDefinition.concept_code.asc()).all()


def upsert_concept_definition(
    db: Session,
    user: User,
    *,
    concept_code: str,
    name: str,
    description: str | None = None,
    parent_code: str | None = None,
    synonyms: list[str] | None = None,
    active: bool = True,
) -> ConceptDefinition:
    code = (concept_code or "").strip()
    if not code:
        raise ValueError("concept_code is required")

    row = (
        db.query(ConceptDefinition)
        .filter(ConceptDefinition.tenant_id == user.id, ConceptDefinition.concept_code == code)
        .first()
    )
    if row:
        row.name = (name or "").strip() or row.name
        row.description = description
        row.parent_code = parent_code
        row.synonyms_json = synonyms or []
        row.active = bool(active)
        row.updated_at = _now_utc()
    else:
        row = ConceptDefinition(
            tenant_id=user.id,
            concept_code=code,
            name=(name or "").strip() or code,
            description=description,
            parent_code=parent_code,
            synonyms_json=synonyms or [],
            active=bool(active),
            created_at=_now_utc(),
            updated_at=_now_utc(),
        )
        db.add(row)

    db.commit()
    db.refresh(row)
    logger.info("CONCEPT_DEFINITION_UPSERT tenant_id=%s code=%s active=%s", user.id, row.concept_code, row.active)
    return row


def create_concept_batch(
    db: Session,
    user: User,
    *,
    model_version: str | None = None,
    summary: dict[str, Any] | None = None,
) -> ConceptBatch:
    row = ConceptBatch(
        tenant_id=user.id,
        model_version=model_version,
        summary_json=summary or {},
        created_at=_now_utc(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("CONCEPT_BATCH_CREATED tenant_id=%s batch_id=%s", user.id, row.id)
    return row


def list_concept_assignments(
    db: Session,
    user: User,
    *,
    concept_code: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    batch_id: str | None = None,
    limit: int = 200,
) -> list[ConceptAssignment]:
    q = db.query(ConceptAssignment).filter(ConceptAssignment.tenant_id == user.id)

    if concept_code:
        q = q.filter(ConceptAssignment.concept_code == concept_code.strip())
    if target_type:
        q = q.filter(ConceptAssignment.target_type == target_type.strip())
    if target_id:
        q = q.filter(ConceptAssignment.target_id == target_id.strip())
    if batch_id:
        q = q.filter(ConceptAssignment.batch_id == batch_id)

    q = q.order_by(ConceptAssignment.updated_at.desc())
    q = q.limit(max(1, min(500, int(limit))))
    return q.all()


def upsert_concept_assignment(
    db: Session,
    user: User,
    *,
    target_type: str,
    target_id: str,
    concept_code: str,
    source: str = "llm_auto",  # llm_auto|manual|import
    confidence: float | None = None,
    rationale: str | None = None,
    evidence_json: dict | None = None,
    locked: bool | None = None,
    batch_id: str | None = None,
) -> tuple[ConceptAssignment, bool]:
    """
    Upsert a concept assignment.

    Manual override rule (MVP):
      - If an existing row is locked OR has source='manual', and you attempt to write source!='manual',
        we DO NOT overwrite it.

    Returns:
      (row, applied) where applied=False means it was blocked by a manual lock.
    """
    tt = (target_type or "").strip()
    tid = (target_id or "").strip()
    cc = (concept_code or "").strip()
    src = (source or "llm_auto").strip()

    if not tt or not tid or not cc:
        raise ValueError("target_type, target_id, concept_code are required")

    row = (
        db.query(ConceptAssignment)
        .filter(
            ConceptAssignment.tenant_id == user.id,
            ConceptAssignment.target_type == tt,
            ConceptAssignment.target_id == tid,
            ConceptAssignment.concept_code == cc,
        )
        .first()
    )

    incoming_is_manual = (src == "manual")

    if row:
        if (bool(row.locked) or (str(row.source or "") == "manual")) and (not incoming_is_manual):
            return row, False

        row.source = src
        row.confidence = confidence
        row.rationale = rationale
        row.evidence_json = evidence_json
        row.batch_id = batch_id
        if locked is not None:
            row.locked = bool(locked)
        else:
            # manual writes lock by default
            if incoming_is_manual:
                row.locked = True
        row.updated_at = _now_utc()
    else:
        row = ConceptAssignment(
            tenant_id=user.id,
            target_type=tt,
            target_id=tid,
            concept_code=cc,
            source=src,
            confidence=confidence,
            rationale=rationale,
            evidence_json=evidence_json,
            locked=(bool(locked) if locked is not None else bool(incoming_is_manual)),
            batch_id=batch_id,
            created_at=_now_utc(),
            updated_at=_now_utc(),
        )
        db.add(row)

    db.commit()
    db.refresh(row)
    logger.info(
        "CONCEPT_ASSIGNMENT_UPSERT tenant_id=%s target=%s:%s concept=%s source=%s locked=%s",
        user.id,
        row.target_type,
        row.target_id,
        row.concept_code,
        row.source,
        row.locked,
    )
    return row, True
