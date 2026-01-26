# models.py
from datetime import datetime
from uuid import uuid4
import enum

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Float,
    Text,
    Index,
    UniqueConstraint,
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    settings = relationship(
        "UserSetting",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # Relationships
    email_accounts = relationship("EmailAccount", back_populates="user")
    # New: user ↔ tasks
    tasks = relationship(
        "Task",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class UserSetting(Base):
    __tablename__ = "user_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # e.g. "agent_greeting", "gmail_summary_enabled", "gmail_account_id"
    key = Column(String, nullable=False)
    value = Column(JSONB, nullable=False, default=dict)

    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="settings")


class EmailAccount(Base):
    __tablename__ = "email_accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    provider_type = Column(String, nullable=False)          # "gmail", "imap_custom", etc.
    oauth_provider = Column(String, nullable=True)
    oauth_access_token = Column(Text, nullable=True)
    oauth_refresh_token = Column(Text, nullable=True)
    oauth_expires_at = Column(DateTime, nullable=True)

    imap_host = Column(String, nullable=True)
    imap_port = Column(Integer, nullable=True)
    imap_ssl = Column(Boolean, nullable=True)
    smtp_host = Column(String, nullable=True)
    smtp_port = Column(Integer, nullable=True)
    smtp_ssl = Column(Boolean, nullable=True)
    username = Column(String, nullable=True)
    password_enc = Column(Text, nullable=True)

    email_address = Column(String, nullable=False)
    display_name = Column(String, nullable=True)

    is_primary = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="email_accounts")


# =========================
# Task Engine Models
# =========================

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    COLLECTING_INPUT = "COLLECTING_INPUT"
    READY = "READY"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class TaskType(str, enum.Enum):
    REMINDER = "REMINDER"
    TIMER = "TIMER"
    EMAIL_CHECK = "EMAIL_CHECK"
    NOTE = "NOTE"
    WORKFLOW = "WORKFLOW"
    COUNTING = "COUNTING"


class Task(Base):
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    type = Column(SAEnum(TaskType, name="task_type_enum"), nullable=False)
    status = Column(SAEnum(TaskStatus, name="task_status_enum"), nullable=False, default=TaskStatus.PENDING)

    # Flexible JSON structures used by the task engine
    # {
    #   "required": [...],
    #   "optional": {...},
    #   "collected": {...}
    # }
    inputs = Column(JSONB, nullable=False, default=dict)

    # {
    #   "cursor": "step_name_or_index",
    #   "context": {...},
    #   "history": [...]
    # }
    state = Column(JSONB, nullable=False, default=dict)

    # {
    #   "result": ...,
    #   "error": "..."
    # }
    execution = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="tasks")


# ---------------------------------------------------------------------------
# Caller-scoped TTL cache (Postgres-backed)
# Purpose:
# - Persist short-term "skill results" across calls for the same caller ID.
# - Keyed by (tenant_id, caller_id, skill_key, cache_key_hash) with expires_at.
# - Used as a best-effort cache to avoid re-hitting external APIs for follow-ups.
# ---------------------------------------------------------------------------

class KBDocumentStatus(enum.Enum):
    pending_link = "pending_link"
    pending_upload = "pending_upload"
    uploaded = "uploaded"
    ingesting = "ingesting"
    ready = "ready"
    failed = "failed"


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Tenant scope (for now, tenant == user id)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Who we texted (optional, useful for auditing)
    target_phone = Column(String, nullable=True)

    filename = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)

    storage_bucket = Column(String, nullable=True)
    storage_key = Column(String, nullable=True)

    status = Column(SAEnum(KBDocumentStatus), nullable=False, default=KBDocumentStatus.pending_link)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_kb_documents_tenant_created", "tenant_id", "created_at"),
    )

class CallerSkillCache(Base):
    __tablename__ = "caller_skill_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Tenant scope (for now, tenant == user id)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # E.164 caller id (e.g., +15551234567). Stored as text to preserve formatting.
    caller_id = Column(String, nullable=False)

    # Skill identifier (e.g., "gmail_summary")
    skill_key = Column(String, nullable=False)

    # Hash of meaningful inputs (e.g., account_id + query + max_results)
    cache_key_hash = Column(String, nullable=False)

    # Cached payload (skill output + any metadata required for follow-ups)
    result_json = Column(JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint("tenant_id", "caller_id", "skill_key", "cache_key_hash", name="uq_caller_skill_cache"),
        Index("ix_caller_skill_cache_lookup", "tenant_id", "caller_id", "skill_key", "cache_key_hash"),
        Index("ix_caller_skill_cache_expires", "expires_at"),
    )



# =========================
# Long-term Memory Events
# =========================

from sqlalchemy import Column, String, DateTime, Text, JSON, Index
from datetime import datetime as _dt

class CallerMemoryEvent(Base):
    __tablename__ = "caller_memory_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    tenant_id = Column(String, index=True, nullable=False)
    caller_id = Column(String, index=True, nullable=False)
    call_sid = Column(String, index=True, nullable=True)

    # kind is required by the existing DB schema (NOT NULL). Used for quick filtering.
    # Values: "turn" for conversational turns, "skill" for skill outputs, "event" fallback.
    kind = Column(String, index=True, nullable=False, default="event")

    created_at = Column(DateTime, default=_dt.utcnow, index=True, nullable=False)

    skill_key = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)

    data_json = Column(JSON, nullable=True)
    tags_json = Column(JSON, nullable=True)

# Helpful composite indexes for time-scoped recall
Index("ix_mem_tenant_caller_created", CallerMemoryEvent.tenant_id, CallerMemoryEvent.caller_id, CallerMemoryEvent.created_at.desc())
Index("ix_mem_tenant_caller_skill_created", CallerMemoryEvent.tenant_id, CallerMemoryEvent.caller_id, CallerMemoryEvent.skill_key, CallerMemoryEvent.created_at.desc())

# =========================
# Analytics Events (Metrics Layer)
# =========================

class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    tenant_id = Column(String, index=True, nullable=False)

    # Optional context (present for call/voice events)
    caller_id = Column(String, index=True, nullable=True)
    call_sid = Column(String, index=True, nullable=True)

    # e.g. "skill_requested", "skill_executed", "schedule_fired", "notification_sent"
    event_type = Column(String, index=True, nullable=False)
    skill_key = Column(String, index=True, nullable=True)

    created_at = Column(DateTime, default=_dt.utcnow, index=True, nullable=False)

    payload_json = Column(JSON, nullable=True)
    tags_json = Column(JSON, nullable=True)

Index("ix_analytics_tenant_type_created", AnalyticsEvent.tenant_id, AnalyticsEvent.event_type, AnalyticsEvent.created_at.desc())
Index("ix_analytics_tenant_skill_created", AnalyticsEvent.tenant_id, AnalyticsEvent.skill_key, AnalyticsEvent.created_at.desc())
Index("ix_analytics_tenant_call_created", AnalyticsEvent.tenant_id, AnalyticsEvent.call_sid, AnalyticsEvent.created_at.desc())


# =========================
# Web Search Skills + Scheduling
# =========================

class DeliveryChannel(enum.Enum):
    sms = "sms"
    whatsapp = "whatsapp"
    email = "email"
    call = "call"


class DeliveryCadence(enum.Enum):
    daily = "daily"


class WebSearchSkill(Base):
    __tablename__ = "web_search_skills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Tenant scope (for now, tenant == user id)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String, nullable=False, default="Web Search")
    query = Column(Text, nullable=False)
    triggers = Column(JSONB, nullable=False, default=list)
    enabled = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_web_search_skills_tenant_created", "tenant_id", "created_at"),
    )



# =========================
# DB Query Skills (generic DB searching skill capability)
# =========================

class DBQuerySkill(Base):
    __tablename__ = "db_query_skills"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Tenant scope (for now, tenant == user id)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String, nullable=False, default="DB Query")
    entity = Column(String, nullable=False, default="caller_memory_events")
    spec = Column(JSONB, nullable=False, default=dict)  # flexible query spec (JSON DSL)
    triggers = Column(JSONB, nullable=False, default=list)
    enabled = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_db_query_skills_tenant_created", "tenant_id", "created_at"),
        Index("ix_db_query_skills_tenant_entity", "tenant_id", "entity"),
    )

class ScheduledDelivery(Base):
    __tablename__ = "scheduled_deliveries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    tenant_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    web_search_skill_id = Column(UUID(as_uuid=True), ForeignKey("web_search_skills.id", ondelete="CASCADE"), nullable=False, index=True)

    enabled = Column(Boolean, nullable=False, default=True)
    cadence = Column(SAEnum(DeliveryCadence), nullable=False, default=DeliveryCadence.daily)

    # "HH:MM" in local timezone
    time_of_day = Column(String, nullable=False, default="08:00")
    timezone = Column(String, nullable=False, default="America/New_York")

    channel = Column(SAEnum(DeliveryChannel), nullable=False, default=DeliveryChannel.sms)
    destination = Column(String, nullable=False)  # phone/email depending on channel

    next_run_at = Column(DateTime, nullable=True, index=True)
    last_run_at = Column(DateTime, nullable=True)
    last_latency_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_scheduled_deliveries_tenant_next", "tenant_id", "next_run_at"),
        Index("ix_scheduled_deliveries_skill", "web_search_skill_id"),
    )