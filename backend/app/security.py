from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import secrets

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash


ALGORITHM = "HS256"
password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return password_hash.verify(password, encoded)
    except Exception:
        return False


def generate_edge_api_key() -> str:
    return secrets.token_urlsafe(32)


def hash_edge_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_edge_api_key(api_key: str, encoded: str) -> bool:
    return secrets.compare_digest(hash_edge_api_key(api_key), encoded)


def create_access_token(user: dict, secret: str, expire_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "driver_id": user.get("driver_id"),
        "ver": int(user.get("token_version") or 0),
        "iat": now,
        "exp": now + timedelta(minutes=expire_minutes),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict:
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp", "iat", "ver"]},
        )
    except InvalidTokenError as error:
        raise ValueError("Token không hợp lệ hoặc đã hết hạn") from error
