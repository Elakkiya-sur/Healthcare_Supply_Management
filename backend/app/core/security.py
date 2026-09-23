"""
Self-contained auth primitives: PBKDF2 password hashing + HS256 JWT.
No external crypto dependencies — works on any Python 3.9+ environment.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional

from app.core.config import settings

# --------------------------------------------------------------- passwords
_ITERATIONS = 260_000


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


# --------------------------------------------------------------- JWT (HS256)
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_access_token(payload: Dict[str, Any], expires_minutes: Optional[int] = None) -> str:
    exp_min = expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    body = dict(payload)
    body["iat"] = int(time.time())
    body["exp"] = int(time.time()) + exp_min * 60

    header = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64e(json.dumps(body, separators=(",", ":")).encode())
    signing_input = f"{header}.{claims}".encode()
    sig = hmac.new(settings.JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{claims}.{_b64e(sig)}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        header, claims, sig = token.split(".")
        expected = hmac.new(
            settings.JWT_SECRET.encode(), f"{header}.{claims}".encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        body = json.loads(_b64d(claims))
        if body.get("exp", 0) < time.time():
            return None
        return body
    except Exception:
        return None
