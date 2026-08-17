"""Speak the answer back, via Sarvam TTS.

The brief's pipeline is *voice in → answer out*, and until now the second half
stopped at text: the system transcribed speech, retrieved, generated and cited,
then rendered the answer silently. A voice assistant that cannot talk is a
search box with extra steps.

Why this is a server endpoint rather than a browser call:

* The Sarvam key is server-side, for the same reason the STT relay exists —
  Sarvam mints no ephemeral browser credential, so a browser that calls TTS
  directly must carry the account key.
* Answers are short by construction (the prompt enforces factoid-shaped
  responses and generation is capped), so one request per answer is cheap and
  there is nothing to stream.

Why not the browser's own `speechSynthesis`: it exists, it is free, and it is
the wrong choice here. Its Indic voice coverage is inconsistent across browsers
and absent on most desktops, and the organisers' dataset is Indic. Using the
same vendor for both directions keeps the demo's voice consistent and the
language list honest.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import Settings

__all__ = ["TtsError", "synthesize", "SARVAM_TTS_URL", "DEFAULT_SPEAKER"]

log = logging.getLogger("voicerag.api.tts")

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

#: Sarvam's `bulbul:v2` voices. `anushka` is the default for no better reason
#: than that it is clear at speed, which is what a demo needs.
DEFAULT_SPEAKER = "anushka"

#: Sarvam rejects requests above this length outright. Answers are capped well
#: below it by MAX_TOKENS, so this is a guard against a pathological response
#: rather than a routine truncation.
_MAX_CHARS = 1500


class TtsError(RuntimeError):
    """Synthesis failed. Carries a message safe to show a client."""


async def synthesize(
    text: str,
    cfg: Settings,
    *,
    language_code: str = "en-IN",
    speaker: str = DEFAULT_SPEAKER,
    client: Any | None = None,
) -> bytes:
    """Return WAV bytes for ``text``.

    Args:
        text: The answer to speak. Truncated at :data:`_MAX_CHARS`.
        cfg: Settings supplying the Sarvam key and model.
        language_code: BCP-47 target. Sarvam speaks the text as written, so this
            selects pronunciation, not translation.
        speaker: Voice id for the configured model.
        client: Injected ``httpx.AsyncClient``, for tests.

    Raises:
        TtsError: If no credential is configured or the vendor refuses.
    """
    if cfg.sarvam_api_key is None:
        raise TtsError("SARVAM_API_KEY is not configured")

    body = text.strip()
    if not body:
        raise TtsError("nothing to speak")
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS]

    payload = {
        "text": body,
        "target_language_code": language_code,
        "model": cfg.sarvam_tts_model,
        "speaker": speaker,
        "speech_sample_rate": 22050,
    }
    headers = {"api-subscription-key": cfg.sarvam_api_key.get_secret_value()}

    owned = client is None
    http = client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await http.post(SARVAM_TTS_URL, headers=headers, json=payload)
    except Exception as exc:  # noqa: BLE001 - surfaced as TtsError
        raise TtsError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if owned:
            await http.aclose()

    if response.status_code != 200:
        # Never echo the vendor body wholesale; it can carry request context.
        raise TtsError(f"sarvam tts returned {response.status_code}")

    try:
        audios = response.json()["audios"]
    except Exception as exc:  # noqa: BLE001
        raise TtsError("sarvam tts returned an unreadable body") from exc
    if not audios:
        raise TtsError("sarvam tts returned no audio")

    import base64

    return base64.b64decode(audios[0])
