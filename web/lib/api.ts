/**
 * Client for the VoiceRAG backend.
 *
 * Everything here is streaming-first. The perceived latency of a voice assistant
 * is time-to-first-token, not time-to-complete-answer, so the UI must render the
 * first token the instant it exists rather than waiting for a finished response.
 */

/**
 * Base URL for every call to the backend, resolved once at module load.
 *
 * Three cases, and the ordering is what makes a misconfigured deploy fail safe:
 *
 * - **Set to a URL** — use it. Two origins, so CORS and the relay's own origin
 *   check both apply and both have to name it.
 * - **Set to the empty string, or unset in a production build** — same origin,
 *   i.e. relative URLs. This is what the container serves: the API hosts the
 *   static export itself, so `/ask` is on the page's own origin, there is no
 *   CORS to configure and no second hostname to keep in sync.
 * - **Unset under `next dev`** — `http://localhost:8000`, because the dev
 *   server is on :3000 and the API is not.
 *
 * `NODE_ENV` is the discriminator rather than a second public variable because
 * Next inlines it at build time exactly like the `NEXT_PUBLIC_` one. What it
 * buys is the default in the third line: previously a missing variable baked
 * `http://localhost:8000` into the production bundle, so the deployed page
 * called *the visitor's own machine* and failed in a way that looks like the
 * server is down rather than like a build-arg that was not passed.
 */
function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE;
  if (configured) return configured.replace(/\/$/, "");
  if (configured === "") return "";
  return process.env.NODE_ENV === "development" ? "http://localhost:8000" : "";
}

/** Exported so the STT client resolves the relay against the same base. */
export const API_BASE = resolveApiBase();

export interface Citation {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  title?: string;
}

export interface StageTiming {
  name: string;
  ms: number;
}

export interface GuardrailReport {
  input_allowed: boolean;
  input_reason?: string | null;
  abstained: boolean;
  abstain_reason?: string | null;
  abstain_confidence?: number;
  abstain_signals?: Record<string, number>;
  // Nullable, not optional-only: FastAPI serialises "grounding did not run"
  // as JSON null, and a `!== undefined` check let null through as scored.
  grounded?: boolean | null;
  grounding_score?: number | null;
  unsupported_claims?: string[];
}

export interface SpeculationStats {
  launched: number;
  cancelled: number;
  hit: boolean;
  saved_ms: number;
}

export interface RagResponse {
  answer: string;
  citations: Citation[];
  abstained: boolean;
  abstain_reason?: string | null;
  guardrails: GuardrailReport;
  provider: string;
  trace: {
    trace_id: string;
    total_ms: number;
    critical_path_ms: number;
    breakdown: Record<string, number>;
  };
  speculation?: SpeculationStats | null;
}

export interface StreamHandlers {
  onToken: (delta: string) => void;
  onFinal: (res: RagResponse) => void;
  onError: (err: Error) => void;
}

/** Mint a short-lived STT credential. The account key stays server-side. */
export async function mintSttToken(): Promise<{ token: string; expires_in: number }> {
  const r = await fetch(`${API_BASE}/stt/token`, { method: "POST" });
  if (!r.ok) throw new Error(`token mint failed: ${r.status}`);
  return r.json();
}

/**
 * Stream an answer over SSE.
 *
 * We parse the SSE frames by hand rather than using EventSource because
 * EventSource cannot issue a POST, and the question belongs in a body rather
 * than a query string.
 *
 * Returns an abort function; call it when the user asks something new so a
 * superseded generation stops consuming tokens immediately.
 */
export function streamAnswer(
  question: string,
  language: string,
  h: StreamHandlers,
): () => void {
  const ctrl = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, language }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        // The server writes real explanations into error bodies — the rate
        // limiter's 429 says whose quota and how long to wait — and throwing
        // on status alone discarded them for a bare "stream failed: 429".
        const body = await res.json().catch(() => null);
        throw new Error(
          (body as { message?: string } | null)?.message ??
            `stream failed: ${res.status}`,
        );
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        let sep: number;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);

          let event = "message";
          const dataLines: string[] = [];
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
          }
          if (!dataLines.length) continue;
          const data = dataLines.join("\n");

          if (event === "token") {
            h.onToken(JSON.parse(data).delta ?? "");
          } else if (event === "final") {
            h.onFinal(JSON.parse(data) as RagResponse);
          } else if (event === "error") {
            throw new Error(JSON.parse(data).message ?? "stream error");
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") h.onError(e as Error);
    }
  })();

  return () => ctrl.abort();
}

/**
 * Warm the server's retrieval cache from a partial transcript.
 *
 * Fire-and-forget by design. The server embeds the partial, runs the ANN search,
 * and caches the result keyed by the normalised text. When the final transcript
 * arrives moments later it is usually identical or near-identical, so `/ask`
 * finds the work already done and skips retrieval entirely.
 *
 * Errors are swallowed deliberately: a failed speculation is a missed
 * optimisation, never a broken request. The real answer path does not depend on
 * this call succeeding.
 */
export function speculate(partial: string, language: string): void {
  if (partial.trim().split(/\s+/).length < 3) return;
  void fetch(`${API_BASE}/speculate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ partial, language }),
    keepalive: true,
  }).catch(() => {});
}

export interface Stats {
  chunks: number;
  strategy: string;
  embedding_model: string;
  index_ready: boolean;
  recent_latency?: Record<string, number>;
}

export async function fetchStats(): Promise<Stats> {
  const r = await fetch(`${API_BASE}/stats`);
  if (!r.ok) throw new Error(`stats failed: ${r.status}`);
  return r.json();
}

export interface Examples {
  examples: string[];
  unanswerable_example: string | null;
  n_indexed: number;
}

/**
 * Questions the served corpus actually contains answers for.
 *
 * Sampled server-side from the shard's own labelled query set, so the presets
 * cannot drift from whatever index is loaded. Falls back to an empty list
 * rather than throwing: a demo with no example chips is worse than one with
 * them, but far better than one that fails to render.
 */
export async function fetchExamples(n = 6): Promise<Examples> {
  try {
    const r = await fetch(`${API_BASE}/examples?n=${n}`);
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  } catch {
    return { examples: [], unanswerable_example: null, n_indexed: 0 };
  }
}

/**
 * Speak an answer aloud.
 *
 * Called after the text has already rendered, never before: synthesis costs
 * ~1.4 s, and blocking the answer on it would trade the thing we optimised
 * hardest for (time to first token) against an ornament. The user reads the
 * first sentence while the audio is still being made.
 *
 * Returns an object URL the caller must revoke, or null if synthesis was
 * refused — a demo should stay usable when TTS is unavailable, not error.
 */
/**
 * The answer as it should be HEARD, not as it is displayed.
 *
 * The prompt requires inline citation markers, so `answer` legitimately ends in
 * "[1][3]" — and Sarvam TTS was reading them: "…treats neuropathy one three."
 * The markers stay on screen (they are the visible link to the numbered
 * Sources list); only the speech strips them. The second pattern also drops
 * bracketed editorial notes like " [truncated at the latency budget]", which
 * the pipeline appends on a deadline exit and which the first, digits-only
 * pattern would miss. Then it heals the punctuation the removals orphaned.
 */
const CITE_MARKS = /\s*\[\s*\d{1,2}(?:\s*[,–-]\s*\d{1,2})*\s*\]/g;
const BRACKET_NOTES = /\s*\[[^\]]{1,80}\]/g;

export function speakable(text: string): string {
  return text
    .replace(CITE_MARKS, "")
    .replace(BRACKET_NOTES, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export async function speak(
  text: string,
  languageCode = "en-IN",
  signal?: AbortSignal,
): Promise<string | null> {
  try {
    const r = await fetch(`${API_BASE}/speak`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, language_code: languageCode }),
      signal,
    });
    if (!r.ok) return null;
    return URL.createObjectURL(await r.blob());
  } catch {
    return null;
  }
}
