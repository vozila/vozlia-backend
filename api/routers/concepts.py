"""VOZLIA FILE PURPOSE
Purpose: Admin endpoints for tenant-scoped Concept Codes (definitions + polymorphic assignments).
Hot path: no
Public interfaces: /admin/concepts/definitions, /admin/concepts/assignments
Reads/Writes: concept_definitions, concept_assignments, concept_batches
Feature flags: CONCEPTS_ENABLED (default OFF)
Failure mode: returns HTTP 4xx for validation/auth; never writes without admin key.
Last touched: 2026-02-03 (concept assignment idempotency + safe errors)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from core.logging import logger

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from models import ConceptDefinition, ConceptAssignment


def _concepts_enabled() -> bool:
    return (os.getenv("CONCEPTS_ENABLED", "0") or "0").strip() == "1"


router = APIRouter(prefix="/admin/concepts", tags=["admin-concepts"], dependencies=[Depends(require_admin_key)])


class ConceptDefinitionIn(BaseModel):
    concept_code: str = Field(..., min_length=2, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    parent_code: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    active: bool = True


class ConceptDefinitionOut(BaseModel):
    id: str
    tenant_id: str
    concept_code: str
    name: str
    description: Optional[str]
    parent_code: Optional[str]
    synonyms: List[str]
    active: bool
    created_at: str
    updated_at: str


class ConceptAssignmentIn(BaseModel):
    target_type: str = Field(..., min_length=1, max_length=80)  # e.g. kb_file, kb_chunk
    target_id: str = Field(..., min_length=1, max_length=200)
    concept_code: str = Field(..., min_length=2, max_length=200)
    source: str = Field("llm_auto", min_length=1, max_length=40)  # llm_auto|manual|import
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    evidence_json: Dict[str, Any] = Field(default_factory=dict)
    locked: Optional[bool] = None
    batch_id: Optional[str] = None


class ConceptAssignmentOut(BaseModel):
    id: str
    tenant_id: str
    target_type: str
    target_id: str
    concept_code: str
    source: str
    confidence: Optional[float]
    rationale: Optional[str]
    evidence_json: Dict[str, Any]
    locked: bool
    batch_id: Optional[str]
    created_at: str
    updated_at: str


@router.post("/definitions", response_model=ConceptDefinitionOut)
def upsert_definition(
    payload: ConceptDefinitionIn,
    db: Session = Depends(get_db),
):
    if not _concepts_enabled():
        raise HTTPException(status_code=404, detail="Concepts disabled")

    user = get_or_create_primary_user(db)
    code = payload.concept_code.strip()

    row = (
        db.query(ConceptDefinition)
        .filter(ConceptDefinition.tenant_id == user.id, ConceptDefinition.concept_code == code)
        .first()
    )
    if row:
        row.name = payload.name.strip()
        row.description = payload.description
        row.parent_code = payload.parent_code
        row.synonyms = payload.synonyms or []
        row.active = bool(payload.active)
        row.updated_at = datetime.utcnow()
    else:
        row = ConceptDefinition(
            tenant_id=user.id,
            concept_code=code,
            name=payload.name.strip(),
            description=payload.description,
            parent_code=payload.parent_code,
            synonyms=payload.synonyms or [],
            active=bool(payload.active),
        )
        db.add(row)

    db.commit()
    db.refresh(row)

    return ConceptDefinitionOut(
        id=str(row.id),
        tenant_id=str(row.tenant_id),
        concept_code=row.concept_code,
        name=row.name,
        description=row.description,
        parent_code=row.parent_code,
        synonyms=list(row.synonyms or []),
        active=bool(row.active),
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )


@router.get("/assignments", response_model=List[ConceptAssignmentOut])
def list_assignments(
    concept_code: Optional[str] = Query(None),
    target_type: Optional[str] = Query(None),
    target_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    if not _concepts_enabled():
        raise HTTPException(status_code=404, detail="Concepts disabled")

    user = get_or_create_primary_user(db)

    q = db.query(ConceptAssignment).filter(ConceptAssignment.tenant_id == user.id)

    if concept_code:
        q = q.filter(ConceptAssignment.concept_code == concept_code.strip())
    if target_type:
        q = q.filter(ConceptAssignment.target_type == target_type.strip())
    if target_id:
        q = q.filter(ConceptAssignment.target_id == target_id.strip())

    rows = q.order_by(ConceptAssignment.created_at.desc()).limit(limit).all()

    out: List[ConceptAssignmentOut] = []
    for r in rows:
        out.append(
            ConceptAssignmentOut(
                id=str(r.id),
                tenant_id=str(r.tenant_id),
                target_type=r.target_type,
                target_id=r.target_id,
                concept_code=r.concept_code,
                source=r.source,
                confidence=r.confidence,
                rationale=r.rationale,
                evidence_json=dict(r.evidence_json or {}),
                locked=bool(r.locked),
                batch_id=str(r.batch_id) if r.batch_id else None,
                created_at=r.created_at.isoformat(),
                updated_at=r.updated_at.isoformat(),
            )
        )
    return out


@router.post("/assignments", response_model=ConceptAssignmentOut)
def create_assignment(
    payload: ConceptAssignmentIn,
    db: Session = Depends(get_db),
):
    if not _concepts_enabled():
        raise HTTPException(status_code=404, detail="Concepts disabled")

    user = get_or_create_primary_user(db)

    # Require the concept definition to exist (keeps concept_code stable/auditable)
    code = payload.concept_code.strip()

    batch_uuid = None
    if payload.batch_id:
        try:
            from uuid import UUID
            batch_uuid = UUID(payload.batch_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid batch_id (must be UUID)")

    d = (
        db.query(ConceptDefinition)
        .filter(ConceptDefinition.tenant_id == user.id, ConceptDefinition.concept_code == code)
        .first()
    )
    if not d:
        raise HTTPException(status_code=404, detail=f"Concept definition not found: {code}")

    target_type = payload.target_type.strip()
    target_id = payload.target_id.strip()
    source = payload.source.strip()

    # Idempotency: if an identical assignment already exists, return it.
    # Manual overrides always win:
    # - manual POST upgrades an existing llm_auto assignment to manual+locked
    # - llm_auto POST never overwrites locked/manual assignments
    existing = (
        db.query(ConceptAssignment)
        .filter(
            ConceptAssignment.tenant_id == user.id,
            ConceptAssignment.target_type == target_type,
            ConceptAssignment.target_id == target_id,
            ConceptAssignment.concept_code == code,
        )
        .first()
    )
    if existing:
        upgraded = False
        if source == "manual":
            if existing.source != "manual":
                existing.source = "manual"
                upgraded = True
            if not existing.locked:
                existing.locked = True
                upgraded = True

            # If caller provides rationale/evidence, merge it (manual edits are auditable).
            if payload.rationale is not None:
                existing.rationale = payload.rationale
                upgraded = True
            if payload.evidence_json:
                merged = dict(existing.evidence_json or {})
                merged.update(payload.evidence_json or {})
                existing.evidence_json = merged
                upgraded = True

            if upgraded:
                existing.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing)

        elif bool(existing.locked):
            # Locked/manual assignment exists; do not overwrite.
            pass
        else:
            # Non-locked assignment exists; keep idempotent and return as-is.
            pass

        logger.info(
            "CONCEPT_ASSIGNMENT_EXISTS tenant_id=%s target_type=%s target_id=%s concept_code=%s source_in=%s source_existing=%s locked=%s upgraded=%s",
            str(existing.tenant_id),
            existing.target_type,
            existing.target_id,
            existing.concept_code,
            source,
            existing.source,
            bool(existing.locked),
            upgraded,
        )

        return ConceptAssignmentOut(
            id=str(existing.id),
            tenant_id=str(existing.tenant_id),
            target_type=existing.target_type,
            target_id=existing.target_id,
            concept_code=existing.concept_code,
            source=existing.source,
            confidence=existing.confidence,
            rationale=existing.rationale,
            evidence_json=dict(existing.evidence_json or {}),
            locked=bool(existing.locked),
            batch_id=str(existing.batch_id) if existing.batch_id else None,
            created_at=existing.created_at.isoformat(),
            updated_at=existing.updated_at.isoformat(),
        )

    locked = bool(payload.locked) if payload.locked is not None else (source == "manual")

    row = ConceptAssignment(
        tenant_id=user.id,
        target_type=target_type,
        target_id=target_id,
        concept_code=code,
        source=source,
        confidence=payload.confidence,
        rationale=payload.rationale,
        evidence_json=payload.evidence_json or {},
        locked=locked,
        batch_id=batch_uuid,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # Race-safe idempotency: if we lost a race to create, fetch and return the winner.
        db.rollback()
        winner = (
            db.query(ConceptAssignment)
            .filter(
                ConceptAssignment.tenant_id == user.id,
                ConceptAssignment.target_type == target_type,
                ConceptAssignment.target_id == target_id,
                ConceptAssignment.concept_code == code,
            )
            .first()
        )
        if winner:
            logger.info(
                "CONCEPT_ASSIGNMENT_RACE tenant_id=%s target_type=%s target_id=%s concept_code=%s",
                str(winner.tenant_id),
                winner.target_type,
                winner.target_id,
                winner.concept_code,
            )
            return ConceptAssignmentOut(
                id=str(winner.id),
                tenant_id=str(winner.tenant_id),
                target_type=winner.target_type,
                target_id=winner.target_id,
                concept_code=winner.concept_code,
                source=winner.source,
                confidence=winner.confidence,
                rationale=winner.rationale,
                evidence_json=dict(winner.evidence_json or {}),
                locked=bool(winner.locked),
                batch_id=str(winner.batch_id) if winner.batch_id else None,
                created_at=winner.created_at.isoformat(),
                updated_at=winner.updated_at.isoformat(),
            )
        raise HTTPException(status_code=409, detail="Concept assignment already exists")
    except Exception:
        db.rollback()
        logger.exception(
            "CONCEPT_ASSIGNMENT_CREATE_FAILED tenant_id=%s target_type=%s target_id=%s concept_code=%s",
            str(user.id),
            target_type,
            target_id,
            code,
        )
        raise HTTPException(status_code=400, detail="Failed to create assignment")

    db.refresh(row)

    return ConceptAssignmentOut(
        id=str(row.id),
        tenant_id=str(row.tenant_id),
        target_type=row.target_type,
        target_id=row.target_id,
        concept_code=row.concept_code,
        source=row.source,
        confidence=row.confidence,
        rationale=row.rationale,
        evidence_json=dict(row.evidence_json or {}),
        locked=bool(row.locked),
        batch_id=str(row.batch_id) if row.batch_id else None,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )

