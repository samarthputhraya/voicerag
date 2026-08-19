# VoiceRAG on AWS EC2 — the path that actually shipped

This is how the live demo runs. It was written after deploying it, not before,
so the failures below are ones that actually happened rather than ones that
seemed likely.

**Result:** `t4g.small` in `ap-south-1` (Mumbai), free for 750 hours/month
through **31 Dec 2026** — more than 24/7 for a month. Public HTTPS with a real
certificate, WebSockets working, ~486 ms answers.

## Why AWS rather than Oracle

Oracle Cloud Always Free is the better *machine* — 12 GB of ARM against 2 GB,
and free permanently rather than until December. It is also the one that failed:
its signup performs a strict card-validity check and rejects many Indian debit
cards outright, with a generic "an error occurred while creating your account"
that no amount of retrying fixes. AWS takes a ₹2 refundable authorisation
instead and accepted the same card.

Keep `deploy/oracle/` — everything in it except the runbook is provider-neutral,
and this guide reuses it verbatim.

## Prerequisites

1. **The image must already be published.** Run the `ARM64 image` workflow
   (`workflow_dispatch`) and make the GHCR package **public**. The VM pulls
   `ghcr.io/<you>/voicerag:demo`; it does not build. This is what turns a
   30-50 minute per-host build into a 3 minute download, and it is why trying a
   second provider costs almost nothing.
2. **A hostname**, free at [duckdns.org](https://www.duckdns.org) — sign in with
   GitHub, create a subdomain. Not sslip.io; see `deploy/oracle/README.md`.

## Launch

EC2 → Launch instances, in **`ap-south-1`**:

| Field | Value | Why |
|---|---|---|
| AMI | **Ubuntu 24.04 LTS**, architecture **64-bit (Arm)** | the image is `arm64`; an x86 instance cannot run it at all |
| Type | **`t4g.small`** | 750 free hours/month to 31 Dec 2026 |
| Storage | **30 GiB** gp3 | free-tier ceiling; the default 8 GiB fills while unpacking |
| Firewall | tick **SSH**, **HTTP**, **HTTPS** | port 80 is needed even for an HTTPS site — Let's Encrypt validates over it |
| Advanced → **Credit specification** | **`Standard`** | ⚠️ the default `Unlimited` silently buys surplus CPU credits, which the free trial does **not** cover. This is the only setting on the page that can generate a bill |
| Advanced → **User data** | paste `deploy/oracle/cloud-init.yaml` | |

Then allocate an **Elastic IP** and associate it. EC2 public IPs are dynamic: a
stop/start changes the address and a submitted link dies with no warning. An
Elastic IP is free *while attached to a running instance* — release it if you
ever terminate.

Point the DuckDNS record at the Elastic IP.

## Configure and deploy

`cloud-init.yaml` installs Docker, clones the repo and copies `env.template`.
Fill in the secrets and run the deploy:

```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
cloud-init status --wait
nano /opt/voicerag/.env      # SARVAM_API_KEY, GROQ_API_KEY, SITE_ADDRESS
bash /opt/voicerag/repo/deploy/oracle/deploy.sh
```

Use `bash <path>` rather than executing it directly if your checkout predates
the mode fix — the file was committed `644` for a while.

## What actually went wrong, in order

**1. `write_files` aborted the entire deployment.** A generated variant of the
cloud-init added `owner: ubuntu:ubuntu` to a file. `write_files` runs before
cloud-init creates the default user on AWS images, so it failed with
`OSError: Unknown user or group: getpwnam(): name not found: 'ubuntu'` — and
took **every other file in the block** with it, including the `.env` and the
auto-deploy script. `runcmd` still ran, so Docker installed and the repo cloned.

From outside this is indistinguishable from a firewall problem: SSH answers,
80 and 443 refuse. `cloud-init status --long` names it immediately, and is the
first thing to run when a box comes up half-built. Oracle's images ship the
`ubuntu` user already, so this only appears on AWS.

**2. The firewall helper could abort the rest of the run.** The Oracle-specific
`voicerag-open-ports` script ran under `set -eu` and ended with
`netfilter-persistent save`. On an image without that binary it exited non-zero
and cloud-init skipped every remaining step. It is now best-effort throughout —
a firewall convenience must not be able to kill the deployment.

**3. `deploy.sh` was committed mode `644`.** The runbook said to execute it by
path, which fails with permission denied. Fixed with
`git update-index --chmod=+x`.

## Telling the failure modes apart from outside

| symptom | meaning |
|---|---|
| 80/443 **time out** | the Security Group has no rule — a dropped packet |
| 80/443 **connection refused** | the packet arrived and nothing is listening: cloud-init is still running, or it failed. Check `cloud-init status --long` |
| TLS error, HTTP works | certificate not issued — DNS wrong, or port 80 unreachable from outside |
| mic lights, no transcript | the relay refused the WebSocket origin. `docker compose logs voicerag \| grep "refused origin"` |

## Costs, and the date that matters

Free now. Two deadlines:

- **31 Dec 2026** — the `t4g.small` free trial ends and the instance becomes
  ~$12/month. **This is the one that will catch you.**
- **~12 months after signup** — the free-tier allowances for EBS and the public
  IPv4 address end.

Set a **$1 budget alert** (Billing → Budgets). Free tier is genuinely free; the
alert is for the day it stops being.

## Verifying

```bash
curl -s -o /dev/null -w '%{http_code} cert=%{ssl_verify_result}\n' https://<host>/healthz
curl -s https://<host>/healthz | python3 -m json.tool | head -20
```

`ssl_verify_result` must be `0`. It is not cosmetic: browsers expose
`getUserMedia` only in a secure context, so an invalid certificate means the
microphone does not exist and the demo has no voice.
