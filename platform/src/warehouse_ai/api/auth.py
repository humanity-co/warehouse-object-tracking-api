from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import HTTPException, status


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def issue_token(subject: str, secret: str, ttl_minutes: int) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "exp": int((datetime.now(tz=timezone.utc) + timedelta(minutes=ttl_minutes)).timestamp()),
        "iat": int(datetime.now(tz=timezone.utc).timestamp()),
    }
    header_segment = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{header_segment}.{payload_segment}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return f"{header_segment}.{payload_segment}.{_b64url_encode(signature)}"


def verify_token(token: str, secret: str) -> Dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token format") from exc
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        f"{header_segment}.{payload_segment}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    received_signature = _b64url_decode(signature_segment)
    if not hmac.compare_digest(expected_signature, received_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token signature")
    payload = json.loads(_b64url_decode(payload_segment))
    if int(payload["exp"]) < int(datetime.now(tz=timezone.utc).timestamp()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token expired")
    return payload

