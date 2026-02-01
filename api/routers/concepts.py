# api/routers/concepts.py
"""VOZLIA FILE PURPOSE
Purpose: Admin endpoints for Concept Codes (definitions/assignments/batches) used for deterministic analytics.
Hot path: no (admin/control plane)
Public interfaces: /admin/concepts/*
Reads/Writes: concept_definitions, concept_assignments, concept_batches
Feature flags: CONCEPTS_ENABLED (must be ON to use endpoints)
Failure mode: returns 404 when disabled; 400/422 on invalid inputs.
Last touched: 2026-02-01 (initial admin endpoints)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.concepts_store import (
    concepts_enabled,
    list_concept_definitions,
    upsert_concept_definition,
    create_concept_batch,
    list_concept_assignments,
    upsert_concept_assignment,
)

router = APIRouter(prefix="/admin/concepts", tags=["admin-concepts"], dependencies=[Depends(require_admin_key)])


def _ensure_enabled() -> None:
    if not concepts_enabled():
        # 404 to avoid hinting that the feature exists when disabled.
        raise HTTPException(status_code=404, detail="Concepts feature is disabled (CONCEPTS_ENABLED=0)")


class ConceptDefinitionIn(BaseModel):
    concept_code: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str | None = None
    parent_code: str | None = None
    synonyms: list[str] | None = None
    active: bool = True


class ConceptDefinitionOut(BaseModel):
    id: str
    concept_code: str
    name: str
    description: str | None = None
    parent_code: str | None = None
    synonyms: list[str] = []
    active: bool = True
    created_at: str | None = None
    updated_at: str | None = None


@router.get("/definitions", response_model=list[ConceptDefinitionOut])
def admin_list_definitions(active_only: bool = False, db: Session = Depends(get_db)):
    _ensure_enabled()
    user = get_or_create_primary_user(db)
    rows = list_concept_definitions(db, user, active_only=bool(active_only))
    return [
        {
            "id": str(r.id),
            "concept_code": r.concept_code,
            "name": r.name,
            "description": r.description,
            "parent_code": r.parent_code,
            "synonyms": (r.synonyms_json or []),
            "active": bool(r.active),
            "created_at": (r.created_at.isoformat() if getattr(r, "created_at", None) else None),
            "updated_at": (r.updated_at.isoformat() if getattr(r, "updated_at", None) else None),
        }
        for r in rows
    ]


@router.post("/definitions", response_model=ConceptDefinitionOut)
def admin_upsert_definition(payload: ConceptDefinitionIn, db: Session = Depends(get_db)):
    _ensure_enabled()
    user = get_or_create_primary_user(db)
    row = upsert_concept_definition(
        db,
        user,
        concept_code=payload.concept_code,
        name=payload.name,
        description=payload.description,
        parent_code=payload.parent_code,
        synonyms=payload.synonyms,
        active=payload.active,
    )
    return {
        "id": str(row.id),
        "concept_code": row.concept_code,
        "name": row.name,
        "description": row.description,
        "parent_code": row.parent_code,
        "synonyms": (row.synonyms_json or []),
        "active": bool(row.active),
        "created_at": (row.created_at.isoformat() if getattr(row, "created_at", None) else None),
        "updated_at": (row.updated_at.isoformat() if getattr(row, "updated_at", None) else None),
    }


class ConceptBatchIn(BaseModel):
    model_version: str | None = None
    summary: dict | None = None


class ConceptBatchOut(BaseModel):
    id: str
    model_version: str | None = None
    created_at: str | None = None
    notified_at: str | None = None
    summary: dict = {}


@router.post("/batches", response_model=ConceptBatchOut)
def admin_create_batch(payload: ConceptBatchIn, db: Session = Depends(get_db)):
    _ensure_enabled()
    user = get_or_create_primary_user(db)
    row = create_concept_batch(db, user, model_version=payload.model_version, summary=payload.summary)
    return {
        "id": str(row.id),
        "model_version": row.model_version,
        "created_at": (row.created_at.isoformat() if getattr(row, "created_at", None) else None),
        "notified_at": (row.notified_at.isoformat() if getattr(row, "notified_at", None) else None),
        "summary": (row.summary_json or {}),
    }


class ConceptAssignmentIn(BaseModel):
    target_type: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    concept_code: str = Field(..., min_length=1)
    source: str = Field(default="manual", min_length=1)  # llm_auto|manual|import
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    evidence_json: dict | None = None
    locked: bool | None = None
    batch_id: str | None = None


class ConceptAssignmentOut(BaseModel):
    id: str
    target_type: str
    target_id: str
    concept_code: str
    source: str
    confidence: float | None = None
    locked: bool
    batch_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    rationale: str | None = None
    evidence_json: dict | None = None


@router.get("/assignments", response_model=list[ConceptAssignmentOut])
def admin_list_assignments(
    concept_code: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    batch_id: str | None = None,
    limit: int = Query(200, ge=1, le=500),
    db: Session = Depends(get_db),
):
    _ensure_enabled()
    user = get_or_create_primary_user(db)
    rows = list_concept_assignments(
        db,
        user,
        concept_code=concept_code,
        target_type=target_type,
        target_id=target_id,
        batch_id=batch_id,
        limit=limit,
    )
    return [
        {
            "id": str(r.id),
            "target_type": r.target_type,
            "target_id": r.target_id,
            "concept_code": r.concept_code,
            "source": str(r.source),
            "confidence": r.confidence,
            "locked": bool(r.locked),
            "batch_id": (str(r.batch_id) if getattr(r, "batch_id", None) else None),
            "created_at": (r.created_at.isoformat() if getattr(r, "created_at", None) else None),
            "updated_at": (r.updated_at.isoformat() if getattr(r, "updated_at", None) else None),
            "rationale": r.rationale,
            "evidence_json": (r.evidence_json or None),
        }
        for r in rows
    ]


@router.post("/assignments", response_model=ConceptAssignmentOut)
def admin_upsert_assignment(payload: ConceptAssignmentIn, db: Session = Depends(get_db)):
    _ensure_enabled()
    user = get_or_create_primary_user(db)
    row, applied = upsert_concept_assignment(
        db,
        user,
        target_type=payload.target_type,
        target_id=payload.target_id,
        concept_code=payload.concept_code,
        source=payload.source,
        confidence=payload.confidence,
        rationale=payload.rationale,
        evidence_json=payload.evidence_json,
        locked=payload.locked,
        batch_id=payload.batch_id,
    )
    if not applied:
        # Manual override blocked an auto-update.
        raise HTTPException(status_code=409, detail="Assignment is locked by manual override")

    return {
        "id": str(row.id),
        "target_type": row.target_type,
        "target_id": row.target_id,
        "concept_code": row.concept_code,
        "source": str(row.source),
        "confidence": row.confidence,
        "locked": bool(row.locked),
        "batch_id": (str(row.batch_id) if getattr(row, "batch_id", None) else None),
        "created_at": (row.created_at.isoformat() if getattr(row, "created_at", None) else None),
        "updated_at": (row.updated_at.isoformat() if getattr(row, "updated_at", None) else None),
        "rationale": row.rationale,
        "evidence_json": (row.evidence_json or None),
    }
