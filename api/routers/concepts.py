"""VOZLIA FILE PURPOSE
Purpose: Admin endpoints for Concept Codes (definitions + assignments) used for deterministic DBQuery filtering.
Hot path: no
Public interfaces: /admin/concepts/definitions, /admin/concepts/assignments
Reads/Writes: concept_definitions, concept_assignments, concept_batches (shared Postgres)
Feature flags: CONCEPTS_ENABLED (when off, endpoints still work but DBQuery has_concept is rejected)
Failure mode: 4xx on validation / missing concept; 5xx on DB errors.
Last touched: 2026-02-02 (initial concepts admin endpoints)
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from models import ConceptAssignment, ConceptBatch, ConceptDefinition, ConceptSource


_CONCEPT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def _concepts_enabled() -> bool:
    v = (os.getenv("CONCEPTS_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _norm_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        raise ValueError("concept_code is required")
    # canonical: lower-case to avoid duplicates like Menu.Steak vs menu.steak
    c = c.lower()
    if not _CONCEPT_CODE_RE.match(c):
        raise ValueError("Invalid concept_code format. Example: menu.steak")
    return c


router = APIRouter(
    prefix="/admin/concepts",
    tags=["admin-concepts"],
    dependencies=[Depends(require_admin_key)],
)


class ConceptDefinitionIn(BaseModel):
    concept_code: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str | None = None
    parent_code: str | None = None
    synonyms: list[str] | None = None
    active: bool = True


class ConceptDefinitionOut(BaseModel):
    id: str
    tenant_id: str
    concept_code: str
    name: str
    description: str | None = None
    parent_code: str | None = None
    synonyms: list[str] = []
    active: bool
    created_at: datetime
    updated_at: datetime


class ConceptAssignmentIn(BaseModel):
    target_type: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    concept_code: str = Field(..., min_length=1)

    source: str = Field(default="manual")  # manual|llm_auto|import
    confidence: float | None = None
    rationale: str | None = None
    evidence_json: dict | None = None
    locked: bool | None = None
    batch_id: str | None = None


class ConceptAssignmentOut(BaseModel):
    id: str
    tenant_id: str
    target_type: str
    target_id: str
    concept_code: str
    source: str
    confidence: float | None = None
    rationale: str | None = None
    evidence_json: dict = {}
    locked: bool
    batch_id: str | None = None
    created_at: datetime
    updated_at: datetime


@router.get("/health")
def concepts_health():
    return {"ok": True, "concepts_enabled": bool(_concepts_enabled())}


@router.post("/definitions", response_model=ConceptDefinitionOut)
def create_concept_definition(payload: ConceptDefinitionIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    try:
        code = _norm_code(payload.concept_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    existing = (
        db.query(ConceptDefinition)
        .filter(ConceptDefinition.tenant_id == user.id)
        .filter(ConceptDefinition.concept_code == code)
        .first()
    )
    if existing:
        # Upsert semantics for MVP: update mutable fields.
        existing.name = payload.name.strip()
        existing.description = (payload.description or None)
        existing.parent_code = (payload.parent_code or None)
        existing.synonyms_json = payload.synonyms or []
        existing.active = bool(payload.active)
        existing.updated_at = datetime.utcnow()
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return {
            "id": str(existing.id),
            "tenant_id": str(existing.tenant_id),
            "concept_code": existing.concept_code,
            "name": existing.name,
            "description": existing.description,
            "parent_code": existing.parent_code,
            "synonyms": existing.synonyms_json or [],
            "active": bool(existing.active),
            "created_at": existing.created_at,
            "updated_at": existing.updated_at,
        }

    row = ConceptDefinition(
        tenant_id=user.id,
        concept_code=code,
        name=payload.name.strip(),
        description=(payload.description or None),
        parent_code=(payload.parent_code or None),
        synonyms_json=payload.synonyms or [],
        active=bool(payload.active),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "id": str(row.id),
        "tenant_id": str(row.tenant_id),
        "concept_code": row.concept_code,
        "name": row.name,
        "description": row.description,
        "parent_code": row.parent_code,
        "synonyms": row.synonyms_json or [],
        "active": bool(row.active),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@router.get("/definitions", response_model=list[ConceptDefinitionOut])
def list_concept_definitions(
    q: str | None = Query(default=None, description="Filter by concept_code prefix or substring"),
    db: Session = Depends(get_db),
):
    user = get_or_create_primary_user(db)
    qry = db.query(ConceptDefinition).filter(ConceptDefinition.tenant_id == user.id)
    if q:
        qq = q.strip().lower()
        qry = qry.filter(ConceptDefinition.concept_code.ilike(f"%{qq}%"))
    rows = qry.order_by(ConceptDefinition.concept_code.asc()).limit(200).all()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": str(r.id),
                "tenant_id": str(r.tenant_id),
                "concept_code": r.concept_code,
                "name": r.name,
                "description": r.description,
                "parent_code": r.parent_code,
                "synonyms": r.synonyms_json or [],
                "active": bool(r.active),
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
        )
    return out


@router.post("/assignments", response_model=ConceptAssignmentOut)
def create_concept_assignment(payload: ConceptAssignmentIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    try:
        code = _norm_code(payload.concept_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Ensure concept exists (prevents typos from becoming "phantom" concepts).
    cd = (
        db.query(ConceptDefinition)
        .filter(ConceptDefinition.tenant_id == user.id)
        .filter(ConceptDefinition.concept_code == code)
        .first()
    )
    if not cd:
        raise HTTPException(status_code=400, detail=f"Unknown concept_code '{code}'. Create it first.")

    # Source validation
    s = (payload.source or "manual").strip().lower()
    if s not in ("manual", "llm_auto", "import"):
        raise HTTPException(status_code=400, detail="source must be one of: manual, llm_auto, import")

    src = ConceptSource.manual if s == "manual" else (ConceptSource.llm_auto if s == "llm_auto" else ConceptSource.import_)

    locked = bool(payload.locked) if payload.locked is not None else (src == ConceptSource.manual)

    # Upsert: unique per tenant/target/concept_code (MVP)
    existing = (
        db.query(ConceptAssignment)
        .filter(ConceptAssignment.tenant_id == user.id)
        .filter(ConceptAssignment.target_type == payload.target_type.strip())
        .filter(ConceptAssignment.target_id == payload.target_id.strip())
        .filter(ConceptAssignment.concept_code == code)
        .first()
    )
    if existing:
        # Manual overrides always win; do not allow llm_auto to overwrite locked rows.
        if bool(existing.locked) and src != ConceptSource.manual:
            raise HTTPException(status_code=409, detail="Assignment is locked (manual override).")

        existing.source = src
        existing.confidence = payload.confidence
        existing.rationale = payload.rationale
        existing.evidence_json = payload.evidence_json or {}
        existing.locked = locked
        existing.batch_id = payload.batch_id
        existing.updated_at = datetime.utcnow()
        db.add(existing)
        db.commit()
        db.refresh(existing)
        row = existing
    else:
        row = ConceptAssignment(
            tenant_id=user.id,
            target_type=payload.target_type.strip(),
            target_id=payload.target_id.strip(),
            concept_code=code,
            source=src,
            confidence=payload.confidence,
            rationale=payload.rationale,
            evidence_json=payload.evidence_json or {},
            locked=locked,
            batch_id=payload.batch_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)

    return {
        "id": str(row.id),
        "tenant_id": str(row.tenant_id),
        "target_type": row.target_type,
        "target_id": row.target_id,
        "concept_code": row.concept_code,
        "source": row.source.value if hasattr(row.source, "value") else str(row.source),
        "confidence": row.confidence,
        "rationale": row.rationale,
        "evidence_json": row.evidence_json or {},
        "locked": bool(row.locked),
        "batch_id": str(row.batch_id) if row.batch_id else None,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@router.get("/assignments", response_model=list[ConceptAssignmentOut])
def list_concept_assignments(
    concept_code: str | None = Query(default=None),
    target_type: str | None = Query(default=None),
    target_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    user = get_or_create_primary_user(db)
    qry = db.query(ConceptAssignment).filter(ConceptAssignment.tenant_id == user.id)
    if concept_code:
        try:
            code = _norm_code(concept_code)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        qry = qry.filter(ConceptAssignment.concept_code == code)
    if target_type:
        qry = qry.filter(ConceptAssignment.target_type == target_type.strip())
    if target_id:
        qry = qry.filter(ConceptAssignment.target_id == target_id.strip())

    rows = qry.order_by(ConceptAssignment.created_at.desc()).limit(limit).all()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": str(r.id),
                "tenant_id": str(r.tenant_id),
                "target_type": r.target_type,
                "target_id": r.target_id,
                "concept_code": r.concept_code,
                "source": r.source.value if hasattr(r.source, "value") else str(r.source),
                "confidence": r.confidence,
                "rationale": r.rationale,
                "evidence_json": r.evidence_json or {},
                "locked": bool(r.locked),
                "batch_id": str(r.batch_id) if r.batch_id else None,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
        )
    return out
