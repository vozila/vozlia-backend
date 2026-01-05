# core/kb_tokens.py
from __future__ import annotations

import os
from dataclasses import dataclass
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


@dataclass(frozen=True)
class TokenConfig:
    secret: str
    salt: str = "vozlia-kb-upload"
    ttl_seconds: int = 1800


def load_token_config() -> TokenConfig:
    secret = (os.getenv("KB_TOKEN_SECRET") or "").strip()
    if not secret:
        raise RuntimeError("KB_TOKEN_SECRET is not set")
    ttl = int(os.getenv("KB_UPLOAD_LINK_TTL_SECONDS") or "1800")
    return TokenConfig(secret=secret, ttl_seconds=ttl)


def _serializer(cfg: TokenConfig) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key=cfg.secret, salt=cfg.salt)


def sign_upload_token(payload: dict, *, cfg: TokenConfig | None = None) -> str:
    cfg = cfg or load_token_config()
    return _serializer(cfg).dumps(payload)


class TokenError(Exception):
    pass


def verify_upload_token(token: str, *, cfg: TokenConfig | None = None) -> dict:
    cfg = cfg or load_token_config()
    try:
        return _serializer(cfg).loads(token, max_age=cfg.ttl_seconds)
    except SignatureExpired as e:
        raise TokenError("Token expired") from e
    except BadSignature as e:
        raise TokenError("Invalid token") from e
