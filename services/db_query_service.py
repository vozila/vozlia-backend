"""VOZLIA FILE PURPOSE
Purpose: Deterministic, tenant-scoped DBQuery execution (safe-field registry + bounded filters) used by dbquery_* dynamic skills and scheduled deliveries.
Hot path: no (called from /assistant/route and background workers; do not call from realtime audio frame loops)
Public interfaces: run_db_query(spec, tenant_uuid)
Reads/Writes: reads from whitelisted tables via SQLAlchemy; no writes.
Feature flags: DB_QUERY_MAX_SPOKEN_CHARS (limits summary size)
Failure mode: returns ok=false with spoken_summary on validation/unsupported fields.
Last touched: 2026-02-01 (add deterministic has_concept filter + keep scalarized summaries)
"""

# NOTE (LEGACY / SLATED FOR REMOVAL)
# ---------------------------------
# This module was part of the earlier DBQuery/Wizard experimentation.
# It is kept for backwards compatibility and troubleshooting only.
# The current direction is to route natural-language intent via an LLM
# (schema-validated) and execute deterministically via the skill engines.
#
# When Intent Router V2 is fully cut over, we should remove or archive this
# DBQuery v1 path (do NOT delete until an env-var gated rollback exists).

# services/db_query_service.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from core.logging import logger


# ---------------------------------------------------------------------------
# Query DSL (intentionally generic to avoid programmatic changes when adding filters)
# ---------------------------------------------------------------------------

FilterOp = Literal[
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "contains",
    "icontains",
    "in",
    "between",
    "is_null",
    "not_null",
    "has_concept",
]

AggOp = Literal["count", "count_distinct", "sum", "avg", "min", "max"]

OrderDir = Literal["asc", "desc"]

TimePreset = Literal[
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_7_days",
    "last_30_days",
]


class DBFilter(BaseModel):
    field: str = Field(..., min_length=1)
    op: FilterOp
    value: Any | None = None
    values: list[Any] | None = None


class DBAggregation(BaseModel):
    op: AggOp
    field: str | None = None  # None => count(*)
    as_name: str | None = None


class DBOrderBy(BaseModel):
    field: str = Field(..., min_length=1)
    direction: OrderDir = "desc"


class DBTimeframe(BaseModel):
    preset: TimePreset | None = None
    start: datetime | None = None
    end: datetime | None = None
    timezone: str = "America/New_York"


class DBQuerySpec(BaseModel):
    entity: str = Field(..., min_length=1)
    select: list[str] | None = None  # list rows (fields)
    filters: list[DBFilter] = Field(default_factory=list)
    timeframe: DBTimeframe | None = None
    group_by: list[str] = Field(default_factory=list)
    aggregations: list[DBAggregation] | None = None
    order_by: list[DBOrderBy] = Field(default_factory=list)
    limit: int = Field(default=25, ge=1, le=200)


class DBQueryResult(BaseModel):
    ok: bool = True
    entity: str
    count: int
    rows: list[dict] = Field(default_factory=list)
    aggregates: dict[str, Any] | None = None
    spoken_summary: str = ""


# ---------------------------------------------------------------------------
# Entity registry (safe, tenant-scoped, single-table queries only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _EntityDef:
    name: str
    model: Any
    tenant_field: str
    tenant_is_uuid: bool = True
    created_at_field: str | None = "created_at"
    exclude_fields: Tuple[str, ...] = ()
    concept_target_type: str | None = None


def _entity_registry() -> dict[str, _EntityDef]:
    # Local imports to avoid import cycles
    from models import CallerMemoryEvent, WebSearchSkill, ScheduledDelivery, Task, KBDocument, KBFile, KBChunk

    reg = {
        # Long-term memory events (turns + skills)
        "caller_memory_events": _EntityDef(
            name="caller_memory_events",
            model=CallerMemoryEvent,
            tenant_field="tenant_id",     # stored as string(uuid)
            tenant_is_uuid=False,
            created_at_field="created_at",
            exclude_fields=(),
            concept_target_type="caller_memory_event",
        ),
        # Saved web search skills
        "web_search_skills": _EntityDef(
            name="web_search_skills",
            model=WebSearchSkill,
            tenant_field="tenant_id",
            tenant_is_uuid=True,
            created_at_field="created_at",
            exclude_fields=(),
        ),
        # Schedules (web search)
        "scheduled_deliveries": _EntityDef(
            name="scheduled_deliveries",
            model=ScheduledDelivery,
            tenant_field="tenant_id",
            tenant_is_uuid=True,
            created_at_field="created_at",
            exclude_fields=("destination",),  # destination can be sensitive; use explicit select if needed
        ),
        # Tasks (wizard / background)
        "tasks": _EntityDef(
            name="tasks",
            model=Task,
            tenant_field="user_id",
            tenant_is_uuid=True,
            created_at_field="created_at",
            exclude_fields=(),
        ),
        # KB documents
        "kb_documents": _EntityDef(
            name="kb_documents",
            model=KBDocument,
            tenant_field="tenant_id",
            tenant_is_uuid=True,
            created_at_field="created_at",
            exclude_fields=("storage_key",),  # internal storage path
            concept_target_type="kb_document",
        ),

        # KB files (Control Plane Phase 1)
        "kb_files": _EntityDef(
            name="kb_files",
            model=KBFile,
            tenant_field="tenant_id",  # stored as string(uuid) in control plane
            tenant_is_uuid=False,
            created_at_field="created_at",
            exclude_fields=("storage_key",),  # internal storage path
            concept_target_type="kb_file",
        ),
        # KB text chunks (Control Plane Phase 2)
        "kb_chunks": _EntityDef(
            name="kb_chunks",
            model=KBChunk,
            tenant_field="tenant_id",  # UUID
            tenant_is_uuid=True,
            created_at_field="created_at",
            exclude_fields=(),
            concept_target_type="kb_chunk",
        ),
    }

    # Optional: Concept Code tables (behind CONCEPTS_ENABLED).
    v = (os.getenv("CONCEPTS_ENABLED") or "0").strip().lower()
    if v in ("1", "true", "yes", "on"):
        from models import ConceptDefinition, ConceptAssignment, ConceptBatch
        reg.update({
            "concept_definitions": _EntityDef(
                name="concept_definitions",
                model=ConceptDefinition,
                tenant_field="tenant_id",
                tenant_is_uuid=True,
                created_at_field="created_at",
                exclude_fields=(),
            ),
            "concept_assignments": _EntityDef(
                name="concept_assignments",
                model=ConceptAssignment,
                tenant_field="tenant_id",
                tenant_is_uuid=True,
                created_at_field="created_at",
                exclude_fields=(),
            ),
            "concept_batches": _EntityDef(
                name="concept_batches",
                model=ConceptBatch,
                tenant_field="tenant_id",
                tenant_is_uuid=True,
                created_at_field="created_at",
                exclude_fields=(),
            ),
        })

    return reg


def supported_entities() -> dict[str, dict[str, Any]]:
    """Expose entity + field metadata so wizard/UI can build specs safely."""
    out: dict[str, dict[str, Any]] = {}
    for key, ed in _entity_registry().items():
        cols = [c.name for c in ed.model.__table__.columns]  # type: ignore[attr-defined]
        allowed = [c for c in cols if c not in set(ed.exclude_fields or ())]
        out[key] = {
            "entity": key,
            "fields": sorted(allowed),
            "created_at_field": ed.created_at_field,
            "tenant_field": ed.tenant_field,
            "concept_target_type": ed.concept_target_type,
        }
    return out



def _concepts_enabled() -> bool:
    v = (os.getenv("CONCEPTS_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


_CONCEPT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


def _parse_concept_codes(f: DBFilter) -> list[str]:
    """Extract concept codes from a DBFilter used with op='has_concept'."""
    codes: list[str] = []
    if isinstance(f.values, list) and f.values:
        codes = [str(x) for x in f.values]
    elif isinstance(f.value, str) and f.value.strip():
        codes = [f.value.strip()]
    elif isinstance(f.value, dict):
        # Accept {"concept_code": "menu.steak"} or {"concept_codes": ["menu.steak", ...]}
        if isinstance(f.value.get("concept_codes"), list):
            codes = [str(x) for x in f.value.get("concept_codes") if str(x).strip()]
        elif isinstance(f.value.get("concept_code"), str):
            codes = [str(f.value.get("concept_code")).strip()]
    # Validate
    codes = [c for c in codes if c]
    if not codes:
        raise ValueError("has_concept requires a concept_code (value) or concept_codes (values)")
    bad = [c for c in codes if not _CONCEPT_CODE_RE.match(c)]
    if bad:
        raise ValueError(f"Invalid concept_code(s): {', '.join(bad[:3])}")
    return codes


def _apply_concept_filters(
    *,
    tenant_uuid: str,
    ed: _EntityDef,
    model: Any,
    q,
    concept_filters: list[DBFilter],
    allowed_fields: set[str],
):
    """Apply concept-based filters using a safe EXISTS subquery."""
    if not concept_filters:
        return q
    if not _concepts_enabled():
        raise ValueError("Concept filters are not enabled (set CONCEPTS_ENABLED=1).")
    if not ed.concept_target_type:
        raise ValueError(f"Entity '{ed.name}' does not support concept filters.")

    # Concept tables are tenant-scoped by UUID; coerce tenant id to UUID.
    try:
        from uuid import UUID
        tenant_uuid_obj = UUID(str(tenant_uuid))
    except Exception:
        raise ValueError("Invalid tenant id for concept filters.")

    from sqlalchemy import String as SAString, cast, select
    from models import ConceptAssignment

    for f in concept_filters:
        field = (f.field or "").strip()
        if not field or field not in allowed_fields:
            raise ValueError(f"Unsupported field for has_concept: {field}")
        col = getattr(model, field)

        codes = _parse_concept_codes(f)

        sub = (
            select(1)
            .select_from(ConceptAssignment)
            .where(
                ConceptAssignment.tenant_id == tenant_uuid_obj,
                ConceptAssignment.target_type == ed.concept_target_type,
                ConceptAssignment.target_id == cast(col, SAString),
                ConceptAssignment.concept_code.in_(codes),
            )
        )
        q = q.filter(sub.exists())

    return q

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _resolve_time_window(tf: DBTimeframe, *, now_utc: datetime) -> tuple[datetime | None, datetime | None]:
    tz_name = (tf.timezone or "America/New_York").strip() or "America/New_York"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    # Explicit start/end wins (assume provided in UTC if tzinfo missing)
    if tf.start or tf.end:
        start = tf.start
        end = tf.end
        if start and start.tzinfo is not None:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if end and end.tzinfo is not None:
            end = end.astimezone(timezone.utc).replace(tzinfo=None)
        return (start, end)

    preset = tf.preset
    if not preset:
        return (None, None)

    now_local = now_utc.replace(tzinfo=timezone.utc).astimezone(tz)
    if preset == "today":
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
    elif preset == "yesterday":
        end_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_local = end_local - timedelta(days=1)
    elif preset == "this_week":
        start_local = (now_local - timedelta(days=now_local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=7)
    elif preset == "last_week":
        end_local = (now_local - timedelta(days=now_local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        start_local = end_local - timedelta(days=7)
    elif preset == "this_month":
        start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # next month: add 32 days then normalize
        nm = (start_local + timedelta(days=32)).replace(day=1)
        end_local = nm
    elif preset == "last_7_days":
        end_local = now_local
        start_local = end_local - timedelta(days=7)
    elif preset == "last_30_days":
        end_local = now_local
        start_local = end_local - timedelta(days=30)
    else:
        return (None, None)

    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    return (start_utc, end_utc)


def _apply_filters(model: Any, q, filters: list[DBFilter], allowed_fields: set[str]):
    clauses = []
    for f in filters or []:
        field = (f.field or "").strip()
        if not field or field not in allowed_fields:
            raise ValueError(f"Unsupported field: {field}")
        col = getattr(model, field)

        op = f.op
        if op == "eq":
            clauses.append(col == f.value)
        elif op == "ne":
            clauses.append(col != f.value)
        elif op == "lt":
            clauses.append(col < f.value)
        elif op == "lte":
            clauses.append(col <= f.value)
        elif op == "gt":
            clauses.append(col > f.value)
        elif op == "gte":
            clauses.append(col >= f.value)
        elif op == "contains":
            clauses.append(col.like(f"%{f.value}%"))
        elif op == "icontains":
            clauses.append(col.ilike(f"%{f.value}%"))
        elif op == "in":
            vals = f.values if isinstance(f.values, list) else ([f.value] if f.value is not None else [])
            clauses.append(col.in_(vals))
        elif op == "between":
            vals = f.values if isinstance(f.values, list) else None
            if not vals or len(vals) != 2:
                raise ValueError("between requires values=[low, high]")
            clauses.append(col.between(vals[0], vals[1]))
        elif op == "is_null":
            clauses.append(col.is_(None))
        elif op == "not_null":
            clauses.append(col.is_not(None))
        else:
            raise ValueError(f"Unsupported op: {op}")

    if clauses:
        q = q.filter(and_(*clauses))
    return q


def run_db_query(db: Session, *, tenant_uuid: str, spec: dict | DBQuerySpec) -> DBQueryResult:
    """Run a tenant-scoped DB query from a validated spec (single-table only)."""
    try:
        parsed = spec if isinstance(spec, DBQuerySpec) else DBQuerySpec.model_validate(spec)
    except ValidationError as e:
        return DBQueryResult(ok=False, entity=str((spec or {}).get("entity") if isinstance(spec, dict) else "unknown"), count=0, rows=[], aggregates=None, spoken_summary=f"Invalid query spec: {e}")

    entity_key = parsed.entity.strip()
    reg = _entity_registry()
    if entity_key not in reg:
        return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=f"Unsupported entity: {entity_key}")

    ed = reg[entity_key]
    model = ed.model

    # Determine allowed fields dynamically from SQLAlchemy model columns
    cols = [c.name for c in model.__table__.columns]  # type: ignore[attr-defined]
    allowed_fields = set([c for c in cols if c not in set(ed.exclude_fields or ())])

    # Base query with tenant filter
    # Tenant column type differs across tables (UUID vs string).
    tenant_val = None
    if ed.tenant_is_uuid:
        try:
            from uuid import UUID
            tenant_val = UUID(str(tenant_uuid))
        except Exception:
            tenant_val = str(tenant_uuid)
    else:
        tenant_val = str(tenant_uuid)
    tenant_col = getattr(model, ed.tenant_field)

    q = db.query(model).filter(tenant_col == tenant_val)

    now_utc = _now_utc()

    # Time window
    if parsed.timeframe and ed.created_at_field and parsed.timeframe:
        start, end = _resolve_time_window(parsed.timeframe, now_utc=now_utc)
        if start and ed.created_at_field in allowed_fields:
            q = q.filter(getattr(model, ed.created_at_field) >= start)
        if end and ed.created_at_field in allowed_fields:
            q = q.filter(getattr(model, ed.created_at_field) < end)

    # Filters
    try:
        normal_filters = [f for f in (parsed.filters or []) if f.op != "has_concept"]
        concept_filters = [f for f in (parsed.filters or []) if f.op == "has_concept"]
        q = _apply_filters(model, q, normal_filters, allowed_fields=allowed_fields)
        q = _apply_concept_filters(
            tenant_uuid=str(tenant_uuid),
            ed=ed,
            model=model,
            q=q,
            concept_filters=concept_filters,
            allowed_fields=allowed_fields,
        )
    except Exception as e:
        return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=str(e))

    # Group-by / aggregations
    aggregates: dict[str, Any] | None = None
    rows_out: list[dict] = []

    if parsed.aggregations:
        group_cols = []
        for gb in parsed.group_by or []:
            if gb not in allowed_fields:
                return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=f"Unsupported group_by field: {gb}")
            group_cols.append(getattr(model, gb))

        agg_cols = []
        for a in parsed.aggregations:
            as_name = (a.as_name or "").strip() or f"{a.op}_{a.field or 'all'}"
            if a.op == "count":
                agg_cols.append(func.count().label(as_name))
            elif a.op == "count_distinct":
                if not a.field or a.field not in allowed_fields:
                    return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=f"Unsupported aggregation field: {a.field}")
                agg_cols.append(func.count(getattr(model, a.field).distinct()).label(as_name))
            else:
                if not a.field or a.field not in allowed_fields:
                    return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=f"Unsupported aggregation field: {a.field}")
                col = getattr(model, a.field)
                if a.op == "sum":
                    agg_cols.append(func.sum(col).label(as_name))
                elif a.op == "avg":
                    agg_cols.append(func.avg(col).label(as_name))
                elif a.op == "min":
                    agg_cols.append(func.min(col).label(as_name))
                elif a.op == "max":
                    agg_cols.append(func.max(col).label(as_name))
                else:
                    return DBQueryResult(ok=False, entity=entity_key, count=0, rows=[], aggregates=None, spoken_summary=f"Unsupported aggregation op: {a.op}")

        q2 = db.query(*group_cols, *agg_cols).select_from(model).filter(tenant_col == tenant_val)
        if parsed.timeframe and ed.created_at_field and parsed.timeframe:
            start, end = _resolve_time_window(parsed.timeframe, now_utc=now_utc)
            if start and ed.created_at_field in allowed_fields:
                q2 = q2.filter(getattr(model, ed.created_at_field) >= start)
            if end and ed.created_at_field in allowed_fields:
                q2 = q2.filter(getattr(model, ed.created_at_field) < end)
        q2 = _apply_filters(model, q2, parsed.filters, allowed_fields=allowed_fields)

        if group_cols:
            q2 = q2.group_by(*group_cols)

        # ordering for aggregated queries: if order_by not given, order by first agg desc
        if parsed.order_by:
            for ob in parsed.order_by:
                if ob.field in allowed_fields:
                    col = getattr(model, ob.field)
                    q2 = q2.order_by(col.asc() if ob.direction == "asc" else col.desc())
                else:
                    # allow ordering by agg label
                    for ac in agg_cols:
                        if ac.key == ob.field:
                            q2 = q2.order_by(ac.asc() if ob.direction == "asc" else ac.desc())
                            break
        else:
            if agg_cols:
                q2 = q2.order_by(agg_cols[0].desc())

        q2 = q2.limit(parsed.limit)

        raw_rows = q2.all()
        for r in raw_rows:
            # SQLAlchemy can return tuple() or Row() depending on version/dialect.
            d: dict[str, Any] = {}

            # Preferred: Row has a _mapping we can convert to a dict keyed by labels.
            if hasattr(r, "_mapping"):
                try:
                    m = dict(getattr(r, "_mapping") or {})
                except Exception:
                    m = {}
                # Keep a stable subset (group_by keys then aggregate labels)
                for gb in parsed.group_by or []:
                    if gb in m:
                        d[gb] = m.get(gb)
                for ac in agg_cols:
                    if ac.key in m:
                        d[ac.key] = m.get(ac.key)
                if d:
                    rows_out.append(d)
                    continue

            # Fallback: treat as tuple (group_by values first, then aggs)
            if isinstance(r, tuple):
                idx = 0
                for gb in parsed.group_by or []:
                    d[gb] = r[idx] if idx < len(r) else None
                    idx += 1
                for ac in agg_cols:
                    d[ac.key] = r[idx] if idx < len(r) else None
                    idx += 1
                rows_out.append(d)
            else:
                # Last resort: store as value and let summarizer stringify.
                rows_out.append({"value": r})

        # If no group_by, collapse to aggregates
        if not group_cols and rows_out:
            aggregates = dict(rows_out[0])
            rows_out = []

    else:
        # Row listing
        select_fields = parsed.select or []
        # default: choose a few safe, common fields
        if not select_fields:
            select_fields = [f for f in ("id", ed.created_at_field or "", "caller_id", "skill_key", "status", "name") if f and f in allowed_fields][:5]

        # enforce allowed
        clean_fields = [f for f in select_fields if f in allowed_fields]

        # order_by
        if parsed.order_by:
            for ob in parsed.order_by:
                if ob.field in allowed_fields:
                    col = getattr(model, ob.field)
                    q = q.order_by(col.asc() if ob.direction == "asc" else col.desc())
        else:
            # default newest first if we have created_at
            if ed.created_at_field and ed.created_at_field in allowed_fields:
                q = q.order_by(getattr(model, ed.created_at_field).desc())

        q = q.limit(parsed.limit)
        rows = q.all()

        for row in rows:
            d = {}
            for f in clean_fields:
                v = getattr(row, f, None)
                if isinstance(v, datetime):
                    d[f] = v.isoformat(timespec="seconds")
                else:
                    d[f] = v
            rows_out.append(d)

    # Count: for aggregated queries without rows, use 1; else row length; else full count
    try:
        if aggregates is not None:
            cnt = 1
        else:
            cnt = len(rows_out)
    except Exception:
        cnt = 0

    spoken = _summarize_for_voice(entity_key, rows_out, aggregates)

    return DBQueryResult(ok=True, entity=entity_key, count=int(cnt), rows=rows_out, aggregates=aggregates, spoken_summary=spoken)


def _scalarize_value(v: Any) -> Any:
    """
    Convert annoying single-element containers like (0,) or Row(0) into scalars for human-readable output.
    """
    if v is None:
        return None
    # Tuples/lists like (0,) -> 0
    if isinstance(v, (tuple, list)) and len(v) == 1:
        return _scalarize_value(v[0])
    # SQLAlchemy Row-like objects can be iterable; if they contain 1 value, scalarize.
    if hasattr(v, "_mapping") or hasattr(v, "__iter__"):
        try:
            seq = list(v)  # type: ignore[arg-type]
            if len(seq) == 1:
                return _scalarize_value(seq[0])
        except Exception:
            pass
    return v


def _summarize_for_voice(entity: str, rows: list[dict], aggregates: dict | None) -> str:
    max_chars = int((os.getenv("DB_QUERY_MAX_SPOKEN_CHARS") or os.getenv("MAX_SPOKEN_CHARS") or "900").strip() or "900")

    if aggregates is not None:
        # Simple: speak key metrics
        parts = []
        for k, v in list(aggregates.items())[:6]:
            vv = _scalarize_value(v)
            if isinstance(vv, datetime):
                vv = vv.isoformat(timespec="seconds")
            parts.append(f"{k}: {vv}")
        out = f"Here’s what I found in {entity}: " + ", ".join(parts) + "."
    elif rows:
        # Speak first few rows lightly
        head = rows[:5]
        bulletish = []
        for r in head:
            # pick 2-3 fields
            items = []
            for k, v in list(r.items())[:3]:
                vv = _scalarize_value(v)
                if isinstance(vv, datetime):
                    vv = vv.isoformat(timespec="seconds")
                items.append(f"{k}={vv}")
            bulletish.append("; ".join(items))
        out = f"Here are the latest results from {entity}: " + " | ".join(bulletish)
        if len(rows) > 5:
            out += f" …and {len(rows) - 5} more."
    else:
        out = f"I didn't find any matching records in {entity}."

    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out
