---
title: VoiceRAG
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Voice-enabled RAG over ai4bharat/MSMARCO-XI, with citations and a latency receipt
---

# VoiceRAG — live demo

Voice-enabled retrieval-augmented generation over
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
Submission for HH Goa 2026, Shortlisting Task 2.

**Open the Space and click the microphone.** The page and the API are one
container on one origin: the UI is a static Next.js export served at `/`, and
the endpoint map is at `/api`.

That is a requirement, not packaging tidiness. CORS does not apply to WebSocket
upgrades, so the origin check in `stt_relay` is the only gate on `WS
/stt/stream`; a split deployment whose `CORS_ORIGINS` omits the frontend gives
you a microphone button that lights up, a waveform that moves, and a transcript
that never arrives. From one origin there is nothing left to misconfigure.

Source: <https://github.com/samarthputhraya/voicerag>

## Why this Space and not Render

The served index is 197,511 chunks and measures roughly 740 MB resident.
Render's `starter` plan is 512 MB — an out-of-memory kill on the first request,
not a tight fit — and `standard` is paid. A Space provides 16 GB on CPU Basic at
no hourly cost, so the whole corpus stays in process memory where the latency
numbers were measured.

**On cost, accurately:** the *hardware* is free (CPU Basic, 2 vCPU, 16 GB,
listed at $0/hour), but **creating** a Docker Space is not. Per the Hub docs:
"Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
and require a paid plan to create: PRO for personal accounts, Team or Enterprise
for organizations." An earlier version of this file called the Space "free",
which was true of the running cost and false of the prerequisite.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The demo UI |
| `GET` | `/api` | This map, as JSON |
| `POST` | `/ask` | One question, one grounded answer with citations |
| `POST` | `/ask/stream` | The same, as SSE |
| `WS` | `/stt/stream` | Relay for browser audio to Sarvam realtime STT |
| `POST` | `/speak` | Answer text to `audio/wav` |
| `POST` | `/speculate` | Warm the retrieval cache from a partial transcript |
| `GET` | `/examples` | Questions this corpus can actually answer |
| `GET` | `/healthz` | Index state, provider circuit breakers, live config |
| `GET` | `/stats` | Corpus description and rolling latency percentiles |

Try it:

```bash
curl -s -X POST https://<space>.hf.space/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is a corporation?"}'
```

## Rate limits

This link is public and holds live credentials on a free tier, so the endpoints
that spend a third-party token are limited: 15/min per client and 45/min overall
for `/ask` and `/ask/stream`, 30 and 90 for `/speak`, with looser limits on the
cheap endpoints and a concurrency cap on the relay. Over the limit you get a
`429` with `Retry-After` and the usual error body. `/healthz` is never limited.

Per-client limits key on `X-Forwarded-For`, which a client writes itself, so
they are advisory — the global limits are the ones that actually protect the
quota, because they count requests rather than identities. Current limits and
occupancy are visible under `config.rate_limit` in `/healthz`.

## Secrets this Space needs

Set under **Settings → Variables and secrets**. Names are
`voicerag.config.Settings` field names, upper-cased — there is no `VOICERAG_`
prefix, and a prefixed variable is silently ignored.

| Secret | Required | Purpose |
|---|---|---|
| `SARVAM_API_KEY` | for voice | Speech-to-text, server-side only |
| `GROQ_API_KEY` | for answers | Primary generation |
| `OPENAI_API_KEY` | recommended | Fallback. Groq's free tier caps at 8,000 tokens/min, which a five-passage RAG prompt exhausts in about four questions |
| `STT_SIGNING_KEY` | optional | Stabilises issued STT capabilities across restarts |

Without generation credentials the service still boots, retrieves, and applies
every guardrail — it returns an explicit "no generation provider configured"
refusal rather than a fabricated answer.
