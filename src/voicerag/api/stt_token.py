"""Minting browser credentials for speech-to-text without shipping the API key.

The architecture puts the browser in a direct WebSocket session with Sarvam,
because Sarvam is hosted in India and the RAG API is co-located with the LLM in
US-West: proxying audio would add an India->US->India round trip to every frame.
That decision creates the problem this module solves -- the browser needs *a*
credential, and the account key must never be it. A key in a JavaScript bundle
is a key in every visitor's devtools, and a hackathon demo whose key is public
for three days is a real incident, not a hypothetical one.

Two modes, and the response says which one produced the token, because the
security properties differ and a client should not have to guess:

``ephemeral``
    ``settings.sarvam_token_url`` is configured, so this server POSTs to the
    vendor's mint endpoint with the account key and returns the short-lived
    credential the vendor issues. The account key never leaves the server, and
    the browser holds something the *vendor* will expire.

``capability``
    No vendor mint endpoint is configured. The server issues its own signed,
    expiring capability: ``v1.<payload>.<hmac>``. It is unforgeable without the
    signing key, it expires, and it is bound to a session id -- but it is a
    capability *for this service*, not a Sarvam credential, so a deployment in
    this mode must terminate audio at its own relay rather than let the browser
    dial the vendor directly.

**Not verified from this container:** Sarvam publishes no ephemeral-credential
endpoint that could be confirmed against the live docs from here, which is
exactly why ``sarvam_token_url`` is configuration rather than a hardcoded URL,
and why the mode is reported rather than assumed. Do not read ``capability``
mode as "the browser can talk to Sarvam"; it cannot, and it says so.
"""

from __future__ import annotations

import base64
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ..config import Settings

__all__ = [
    "SttCapability",
    "TokenMintError",
    "issue_capability",
    "verify_capability",
    "mint_token",
]

#: Version prefix, so the format can change without silently accepting an old one.
_VERSION = "v1"


class TokenMintError(RuntimeError):
    """No credential could be issued (no key configured, or the vendor refused)."""


@dataclass(slots=True, frozen=True)
class SttCapability:
    """The decoded contents of a signed capability token.

    Attributes:
        session_id: Random per-token id, usable as a log correlation key and as
            a revocation handle.
        provider: STT vendor the capability is scoped to.
        expires_at: Unix seconds after which the token is rejected.
        issued_at: Unix seconds the token was minted.
    """

    session_id: str
    provider: str
    expires_at: int
    issued_at: int

    @property
    def expires_in(self) -> int:
        """Seconds of validity left; never negative."""
        return max(0, self.expires_at - int(time.time()))


def _b64(raw: bytes) -> str:
    """URL-safe base64 without padding, so the token is query-string clean."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signing_key(settings: Settings) -> bytes:
    """The HMAC key, generated per process when unset.

    A random per-process key is the safe default: it means tokens minted before
    a restart stop working, which is a mild annoyance, rather than that tokens
    are signed with a predictable constant, which is not a defence at all.
    """
    configured = settings.stt_signing_key
    if configured is not None:
        return configured.get_secret_value().encode("utf-8")
    return _EPHEMERAL_KEY


#: Regenerated on import, i.e. once per process. See :func:`_signing_key`.
_EPHEMERAL_KEY = secrets.token_bytes(32)


def issue_capability(settings: Settings, *, provider: str = "sarvam") -> tuple[str, SttCapability]:
    """Sign a short-lived capability for the browser.

    Args:
        settings: Configuration supplying the TTL and the signing key.
        provider: STT vendor the capability is scoped to.

    Returns:
        ``(token, capability)``. The token is ``v1.<payload>.<hmac>``.
    """
    now = int(time.time())
    cap = SttCapability(
        session_id=secrets.token_urlsafe(12),
        provider=provider,
        expires_at=now + settings.stt_token_ttl_s,
        issued_at=now,
    )
    payload = _b64(
        json.dumps(
            {
                "sid": cap.session_id,
                "provider": cap.provider,
                "exp": cap.expires_at,
                "iat": cap.issued_at,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    signature = _b64(hmac.new(_signing_key(settings), payload.encode("ascii"), sha256).digest())
    return f"{_VERSION}.{payload}.{signature}", cap


def verify_capability(token: str, settings: Settings) -> SttCapability:
    """Validate a token issued by :func:`issue_capability`.

    Args:
        token: The token string as handed to the browser.
        settings: Configuration supplying the signing key.

    Returns:
        The decoded :class:`SttCapability`.

    Raises:
        TokenMintError: Malformed, wrongly signed, or expired. One exception
            type for all three, and no detail about which: telling an attacker
            whether the signature or the expiry failed is free information.
    """
    try:
        version, payload, signature = token.split(".")
    except ValueError as exc:
        raise TokenMintError("malformed capability token") from exc
    if version != _VERSION:
        raise TokenMintError("malformed capability token")
    expected = _b64(hmac.new(_signing_key(settings), payload.encode("ascii"), sha256).digest())
    if not hmac.compare_digest(expected, signature):
        raise TokenMintError("malformed capability token")
    try:
        body = json.loads(_unb64(payload))
        cap = SttCapability(
            session_id=str(body["sid"]),
            provider=str(body["provider"]),
            expires_at=int(body["exp"]),
            issued_at=int(body["iat"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise TokenMintError("malformed capability token") from exc
    if cap.expires_at <= int(time.time()):
        raise TokenMintError("malformed capability token")
    return cap


async def mint_token(settings: Settings, *, client: Any | None = None) -> dict[str, Any]:
    """Produce the body of ``POST /stt/token``.

    Args:
        settings: Configuration.
        client: An ``httpx.AsyncClient``-shaped object, injectable for tests.
            Built on demand when a vendor mint endpoint is configured.

    Returns:
        A mapping with ``token``, ``expires_in``, ``provider`` and ``mode``.

    Raises:
        TokenMintError: No STT key is configured, or the vendor refused.
    """
    if settings.sarvam_api_key is None and settings.elevenlabs_api_key is None:
        raise TokenMintError(
            "no STT credential configured; set SARVAM_API_KEY or ELEVENLABS_API_KEY"
        )
    provider = "sarvam" if settings.sarvam_api_key is not None else "elevenlabs"

    if settings.sarvam_token_url and settings.sarvam_api_key is not None:
        body = await _mint_from_vendor(settings, client=client)
        return {
            "token": body["token"],
            "expires_in": int(body.get("expires_in", settings.stt_token_ttl_s)),
            "provider": "sarvam",
            "mode": "ephemeral",
        }

    token, cap = issue_capability(settings, provider=provider)
    return {
        "token": token,
        "expires_in": cap.expires_in,
        "provider": provider,
        "mode": "capability",
    }


async def _mint_from_vendor(settings: Settings, *, client: Any | None) -> dict[str, Any]:
    """POST to the configured vendor endpoint and normalise its answer.

    The vendor's field name for the credential is not fixed across providers, so
    the first of ``token`` / ``access_token`` / ``api_subscription_key`` that is
    present wins. Raises rather than guessing if none is.
    """
    assert settings.sarvam_api_key is not None
    owns_client = client is None
    if client is None:
        import httpx  # imported lazily: the offline path must not need httpx

        client = httpx.AsyncClient(timeout=5.0)
    try:
        response = await client.post(
            settings.sarvam_token_url,
            headers={"api-subscription-key": settings.sarvam_api_key.get_secret_value()},
            json={"expires_in": settings.stt_token_ttl_s},
        )
        if int(response.status_code) >= 400:
            raise TokenMintError(f"vendor refused: HTTP {response.status_code}")
        body = response.json()
    except TokenMintError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalised into one error type
        raise TokenMintError(f"vendor mint failed: {type(exc).__name__}") from exc
    finally:
        if owns_client:
            await client.aclose()

    for key in ("token", "access_token", "api_subscription_key"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return {"token": value, "expires_in": body.get("expires_in")}
    raise TokenMintError("vendor response contained no recognisable credential")
