#!/usr/bin/env bash
# Build and start VoiceRAG on an Oracle Always Free instance.
#
#   tmux new -s deploy
#   /opt/voicerag/repo/deploy/oracle/deploy.sh
#
# Idempotent: safe to re-run after editing .env, after a `git pull`, or after a
# failed attempt. Run it inside tmux — the first build takes 15-25 minutes on
# Ampere because it embeds 197,511 passages, and losing the SSH session in the
# middle of that should not lose the build.

set -euo pipefail

REPO="${REPO:-/opt/voicerag/repo}"
ENV_FILE="${ENV_FILE:-/opt/voicerag/.env}"
COMPOSE_DIR="$REPO/deploy/oracle"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mxx  %s\033[0m\n' "$*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "no $ENV_FILE — copy deploy/oracle/env.template to it first"

# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

# --- hostname ----------------------------------------------------------------
# Caddy needs a name, not an address: Let's Encrypt does not issue for bare IPs.
# sslip.io resolves <ip>.sslip.io to <ip>, which makes a real hostname out of an
# instance that has none, with no domain to buy and no DNS to configure.
if [ -z "${SITE_ADDRESS:-}" ]; then
    say "SITE_ADDRESS is blank; deriving one from the public IP"
    PUBLIC_IP="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
    [ -n "$PUBLIC_IP" ] || die "could not determine the public IP; set SITE_ADDRESS in $ENV_FILE"
    SITE_ADDRESS="${PUBLIC_IP}.sslip.io"
    export SITE_ADDRESS
fi
say "serving at https://${SITE_ADDRESS}"

# Fail early and legibly rather than letting Caddy retry a doomed certificate
# request for ten minutes. A mismatch here is the single most common cause of
# "the site never came up".
RESOLVED="$(getent hosts "$SITE_ADDRESS" | awk '{print $1}' | head -1 || true)"
MYIP="$(curl -fsS --max-time 10 https://api.ipify.org || true)"
if [ -n "$RESOLVED" ] && [ -n "$MYIP" ] && [ "$RESOLVED" != "$MYIP" ]; then
    warn "$SITE_ADDRESS resolves to $RESOLVED but this host is $MYIP."
    warn "Let's Encrypt validates over HTTP to the resolved address, so the"
    warn "certificate will fail until the DNS A record points here."
fi

# --- preflight ---------------------------------------------------------------
[ -n "${SARVAM_API_KEY:-}" ] || warn "SARVAM_API_KEY is empty — the microphone will report stt_unconfigured"
[ -n "${GROQ_API_KEY:-}" ]   || warn "GROQ_API_KEY is empty — every question will refuse with 'no generation provider'"
[ -n "${OPENAI_API_KEY:-}" ] || warn "OPENAI_API_KEY is empty — no fallback when Groq's 8k tok/min runs out"

docker info >/dev/null 2>&1 || die "cannot talk to docker (new group membership? log out and back in)"

for port in 80 443; do
    if ! sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
        warn "iptables is not accepting :$port — opening it"
        sudo iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT
        sudo netfilter-persistent save >/dev/null
    fi
done
say "iptables allows 80 and 443 (the cloud Security List is separate — check it too)"

# --- build -------------------------------------------------------------------
cd "$COMPOSE_DIR"
say "building (15-25 min on Ampere: it embeds 197,511 passages)"
docker compose build

say "starting"
docker compose up -d

# --- wait --------------------------------------------------------------------
# The index load plus the mandatory startup warmup is ~20-40 s on this hardware.
say "waiting for the app to become healthy"
for i in $(seq 1 60); do
    if docker compose exec -T voicerag curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; then
        break
    fi
    sleep 5
    [ "$i" = 60 ] && die "app never became healthy — docker compose logs voicerag"
done

READY="$(docker compose exec -T voicerag curl -fsS http://localhost:8000/healthz)"
echo "$READY" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print()
print("  status          :", h["status"])
print("  chunks          :", h["n_chunks"])
print("  providers       :", ", ".join(h["generation_providers"]) or "NONE")
print("  embedder        :", h["config"].get("embedder_spec"))
print("  rate limit      :", "on" if h["config"]["rate_limit"]["enabled"] else "OFF")
' || echo "$READY"

say "waiting for the certificate (Let's Encrypt, usually under 60 s)"
for i in $(seq 1 40); do
    CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://${SITE_ADDRESS}/healthz" || true)"
    if [ "$CODE" = "200" ]; then
        say "LIVE:  https://${SITE_ADDRESS}"
        echo
        echo "  Open it, click the microphone, and allow access."
        echo "  Endpoint map: https://${SITE_ADDRESS}/api"
        echo
        exit 0
    fi
    sleep 5
done

warn "the app is healthy but https://${SITE_ADDRESS} did not answer 200 yet."
warn "Almost always one of these, in order:"
warn "  1. The cloud Security List has no ingress rule for 80/443 (separate"
warn "     from iptables, and both are required)."
warn "  2. DNS for $SITE_ADDRESS does not point at this instance."
warn "  3. Let's Encrypt is still retrying:  docker compose logs caddy"
exit 1
