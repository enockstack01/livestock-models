"""HS256 JWT verification (stdlib only) for the shared Agricoders login token.

Deliberately minimal and strict: only HS256 is accepted (never 'none', never an algorithm chosen by the token),
the signature is compared in constant time, and an expiry is mandatory.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class AuthError(Exception):
    pass


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def verify_token(token: str, secret: str, now: float | None = None) -> dict:
    if not secret:
        raise AuthError("auth is not configured")
    try:
        h, p, sig = token.split(".")
        header, claims = json.loads(_b64d(h)), json.loads(_b64d(p))
        given = _b64d(sig)
    except Exception:
        raise AuthError("malformed token")
    if header.get("alg") != "HS256":
        raise AuthError("unsupported algorithm")
    want = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(want, given):
        raise AuthError("bad signature")
    t = time.time() if now is None else now
    if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] <= t:
        raise AuthError("token expired")
    if isinstance(claims.get("nbf"), (int, float)) and claims["nbf"] > t:
        raise AuthError("token not yet valid")
    if not claims.get("sub"):
        raise AuthError("token has no subject")
    return claims


def make_token(claims: dict, secret: str) -> str:
    """For tests and local development only (production tokens come from the Agricoders login service)."""
    h = _b64e(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = _b64e(json.dumps(claims).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64e(sig)}"


def dev_secret(path=None) -> str:
    """A random secret kept in ./.dev_jwt_secret so that `serve --dev` on the Vision API and the LLM API share it (one token works on both).
    Local development only: production uses the secret shared with the Agricoders login service."""
    import secrets
    from pathlib import Path
    p = Path(path or Path(__file__).resolve().parents[2] / ".dev_jwt_secret")
    if not p.exists():
        p.write_text(secrets.token_hex(24))
    return p.read_text().strip()
