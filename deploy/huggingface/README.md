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

# VoiceRAG — live API

Voice-enabled retrieval-augmented generation over
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI).
Submission for HH Goa 2026, Shortlisting Task 2.

Source: <https://github.com/samarthputhraya/voicerag>

## Why this Space and not Render

The served index is 197,511 chunks and measures roughly 740 MB resident.
Render's `starter` plan is 512 MB — an out-of-memory kill on the first request,
not a tight fit — and `standard` is paid. A free Space provides 16 GB, so the
whole corpus stays in process memory where the latency numbers were measured.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | One question, one grounded answer with citations |
| `POST` | `/ask/stream` | The same, as SSE |
| `WS` | `/stt/stream` | Relay for browser audio to Sarvam realtime STT |
| `POST` | `/speculate` | Warm the retrieval cache from a partial transcript |
| `GET` | `/healthz` | Index state, provider circuit breakers, live config |
| `GET` | `/stats` | Corpus description and rolling latency percentiles |

Try it:

```bash
curl -s -X POST https://<space>.hf.space/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is a corporation?"}'
```

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
