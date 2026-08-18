# VoiceRAG on Oracle Cloud "Always Free"

The genuinely-$0 deployment: a 4 OCPU / 24 GB Ampere A1 instance, Docker, and
Caddy for automatic HTTPS. One origin, one container for the app, no monthly
bill and no sleep-on-idle.

It costs about an hour of your attention instead, and two things below will
waste an afternoon if you meet them without warning. They are marked.

---

## Why HTTPS is not optional here

`getUserMedia` — the browser API behind the microphone button — only works in a
**secure context**. Over plain HTTP on a bare IP, the mic button does nothing,
and the browser explains why only in the console. A voice demo on `http://` is a
voice demo with no voice.

Managed hosts hand you TLS for free. A raw VM does not, which is the real reason
this path is more work than a Space. Caddy closes the gap: it obtains and renews
a Let's Encrypt certificate by itself, given a *hostname*.

And you get a hostname without buying a domain: **`<your-ip>.sslip.io`** resolves
to the IP it names. `deploy.sh` derives it automatically. If you own a domain,
point an A record at the instance and set `SITE_ADDRESS` instead.

---

## ⚠️ The two things that actually go wrong

**1. "Out of host capacity" on Ampere.** The free ARM shape is heavily
oversubscribed and creation fails with this error in many regions, sometimes for
days. It is the single most likely reason this plan stalls. Mitigations, in
order of how well they work:

- Try each availability domain in your region, one at a time.
- Ask for a smaller shape: **1 OCPU / 6 GB is enough** for this service (the
  index is ~740 MB resident). Smaller requests are filled far more often.
- Retry on a schedule — capacity is released continuously, and early morning in
  the region's local time is materially better.
- Upgrade the account to Pay As You Go. It stays free within the Always Free
  limits, and it moves you out of the trial queue that is deprioritised for ARM.

Do **not** fall back to the free AMD shape (`VM.Standard.E2.1.Micro`). It has
1 GB of RAM; this service needs ~740 MB resident *plus* Python and the build. It
will OOM.

**2. Oracle's images have a second firewall.** The cloud-side **Security List**
and the instance's own **iptables** are independent, and Oracle's Ubuntu images
ship an INPUT chain that drops everything except SSH. Opening only the Security
List leaves Let's Encrypt unable to reach port 80, and the symptom is not
"blocked" — it is a certificate request that quietly times out, which looks like
a Caddy problem. `cloud-init.yaml` and `deploy.sh` both open iptables; **you
must open the Security List yourself** (step 3 below).

---

## Steps

### 1. Create the instance

<https://cloud.oracle.com> → Compute → Instances → **Create instance**.

| Field | Value |
|---|---|
| Image | **Canonical Ubuntu 24.04** (`aarch64` build) |
| Shape | **VM.Standard.A1.Flex**, 4 OCPU / 24 GB (drop to 1/6 if capacity is refused) |
| Networking | Assign a **public IPv4 address** |
| SSH keys | Upload your public key, or let Oracle generate one and **save the private key** |

Then **Show advanced options → Management → User data → Paste cloud-init
script**, and paste the whole of [`cloud-init.yaml`](cloud-init.yaml).

> Ubuntu, not Oracle Linux. The cloud-init installs Docker from Docker's own
> apt repository and uses `iptables-persistent`; on Oracle Linux you would want
> `dnf install docker-ce` and `firewall-cmd --add-service=https --permanent`
> instead. Either works — only one is written down here.

### 2. Wait for cloud-init

```bash
ssh ubuntu@<public-ip>
cloud-init status --wait          # "status: done"
cat /opt/voicerag/READY
```

If Docker commands say "permission denied", log out and back in — group
membership is only picked up on a new login.

### 3. Open the Security List ⚠️

In the console: **Networking → Virtual Cloud Networks → your VCN → Subnet →
Security List → Add Ingress Rules.** Two rules:

| Source CIDR | Protocol | Destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | 80 |
| `0.0.0.0/0` | TCP | 443 |

Port 80 is required even though the site is HTTPS — Let's Encrypt validates over
it.

### 4. Secrets

```bash
nano /opt/voicerag/.env
```

Fill in `SARVAM_API_KEY` and `GROQ_API_KEY` at minimum, and `OPENAI_API_KEY` if
you have one — Groq's free tier is ~8,000 tokens/min, roughly four or five
answers, after which the demo says "Every generation provider failed".

Leave `SITE_ADDRESS` blank to get `<ip>.sslip.io` automatically.

### 5. Deploy

```bash
tmux new -s deploy
/opt/voicerag/repo/deploy/oracle/deploy.sh
```

**Use tmux.** The first build takes 15–25 minutes on Ampere because it downloads
a 440 MB shard and embeds 197,511 passages; losing the SSH session should not
lose the build. Detach with `Ctrl-b d`, reattach with `tmux attach -t deploy`.

The script prints the live URL when the certificate is issued and `/healthz`
answers 200 over TLS.

### 6. Check it before you film

```
https://<site>/            the demo
https://<site>/api         the endpoint map
https://<site>/healthz     index state, providers, rate limits
```

In the browser: click the mic, allow access, ask *"Can gabapentin treat
neuropathy?"*. Then switch the language selector to हिन्दी and ask it again —
partials render in Devanagari, the final transcript arrives in English, and the
answer is spoken back in Hindi.

---

## Redeploying

```bash
cd /opt/voicerag/repo && git pull
deploy/oracle/deploy.sh
```

Idempotent. Certificates live in a named volume, so restarts do not re-request
them — which matters, because five requests in an hour is how you find the
Let's Encrypt rate limit on the morning of a demo.

## Diagnosing

```bash
cd /opt/voicerag/repo/deploy/oracle
docker compose logs -f voicerag       # app
docker compose logs -f caddy          # certificates
docker compose ps
```

| Symptom | Cause, in order of likelihood |
|---|---|
| Site never comes up, cert never issues | Security List rule missing (step 3); then DNS; then iptables |
| Mic lights up, waveform moves, **no transcript ever** | The relay refused the upgrade. `docker compose logs voicerag \| grep "refused origin"`. Caddy sets `X-Forwarded-Host`, which is what the same-origin check reads — this should not happen behind this Caddyfile, and if it does, that header is why |
| Every answer refuses with "no generation provider" | `GROQ_API_KEY` empty or wrong in `/opt/voicerag/.env` |
| "Every generation provider failed" mid-demo | Groq's 8k tok/min exhausted. This is what `OPENAI_API_KEY` is for |
| Build killed around the embedding step | Out of memory on a small shape. The 4 GB swap from cloud-init usually covers it; otherwise use a larger shape |
| `429` responses | The rate limiter, working. See `config.rate_limit` in `/healthz` |

## What this costs

Nothing, within Always Free: the A1 allowance is 4 OCPU and 24 GB of memory
across all your ARM instances, plus 200 GB of block volume and 10 TB/month of
egress. This deployment uses one instance and the boot volume.

Set a **budget alert** anyway (Billing → Budgets, threshold $1). Always Free is
genuinely free, but an accidental second instance or a larger shape is not, and
a $1 alert tells you within a day.
