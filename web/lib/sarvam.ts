/**
 * Sarvam realtime speech-to-text over WebSocket, via our own relay.
 *
 * This file used to dial `wss://api.sarvam.ai` directly, on the reasoning that
 * Sarvam is in India and our API is in US-West, so proxying audio would add a
 * Pacific round trip per frame. The latency argument was sound. The security
 * premise underneath it was not:
 *
 * Sarvam's realtime endpoint accepts **no `token` query parameter** -- auth is
 * an `api-subscription-key` header or subprotocol -- and Sarvam publishes **no
 * ephemeral-token endpoint**. So there is no short-lived credential to mint. The
 * only value that authenticates a browser is the permanent account key, sitting
 * in devtools with no expiry. The old code sent a signed capability our own
 * server had minted, which Sarvam ignored, so the socket died unauthenticated.
 *
 * We therefore relay through `WS /stt/stream`, which holds the key server-side
 * and speaks Sarvam's protocol verbatim in both directions. For the local demo
 * that hop is loopback and free. For a distant deployment it is a real cost, and
 * the alternative is shipping the account key to every visitor.
 */

export type TranscriptEventKind =
  | "session_begin"
  | "speech_start"
  | "speech_end"
  | "partial"
  | "final"
  | "error"
  | "session_end";

export interface TranscriptEvent {
  kind: TranscriptEventKind;
  /** Transcript text for `partial` / `final`, error message for `error`. */
  text: string;
  /** ms since the session opened, from a monotonic clock. */
  tMs: number;
  /** Fatal errors terminate the session; non-fatal ones are informational. */
  fatal?: boolean;
  raw?: unknown;
}

export interface SarvamConfig {
  /**
   * Optional. Kept so a deployment that configures `SARVAM_TOKEN_URL` (or
   * switches to ElevenLabs, which does mint browser credentials) can pass one
   * through. The relay path does not need it and ignores it.
   */
  token?: string;
  /** BCP-47 (e.g. "en-IN", "hi-IN") or "auto" for language identification. */
  languageCode?: string;
  /** "fast" trades a little accuracy for the lowest time-to-first-token. */
  streamType?: "fast" | "balanced";
  /** Silence before Sarvam decides the utterance ended, in ms. */
  silenceDurationMs?: number;
  /** Ignore speech bursts shorter than this, in ms. Suppresses coughs/clicks. */
  minSpeechDurationMs?: number;
  /**
   * "manual" lets our client-side VAD decide the endpoint, which is strictly
   * faster than waiting for the server to notice silence: the browser already
   * knows you stopped talking before the last audio frame finishes uploading.
   */
  endpointing?: "vad" | "manual";
  /**
   * "translate" makes Sarvam return **English** text for speech in any of its
   * supported Indian languages, in the same call and with no latency penalty
   * (measured: 945 ms translating Hindi vs 1006 ms transcribing it).
   *
   * That is what makes this system genuinely multilingual without a
   * multilingual embedder. The corpus is indexed in English, and the query
   * encoder is a model2vec static model whose tokenizer shatters Devanagari
   * into single characters and [UNK] — measured cross-lingual alignment was
   * 1/4, i.e. chance. Translating at the STT boundary sidesteps that entirely:
   * everything downstream of the transcript stays monolingual and fast.
   */
  mode?: "transcribe" | "translate";
  baseUrl?: string;
}

/**
 * The relay endpoint on our own API, derived from the same base URL the REST
 * client uses so a deployment configures one variable, not two. http -> ws and
 * https -> wss, because a page served over TLS cannot open an insecure socket.
 */
function defaultRelayUrl(): string {
  const base =
    process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";
  return `${base.replace(/^http/, "ws")}/stt/stream`;
}

const DEFAULTS = {
  token: "",
  languageCode: "en-IN",
  streamType: "fast" as const,
  silenceDurationMs: 300,
  minSpeechDurationMs: 250,
  endpointing: "vad" as const,
  mode: "transcribe" as const,
  baseUrl: defaultRelayUrl(),
};

/** Float32 PCM in [-1, 1] -> little-endian 16-bit PCM, which is what Sarvam wants. */
export function floatTo16BitPcm(input: Float32Array): ArrayBuffer {
  const out = new DataView(new ArrayBuffer(input.length * 2));
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out.buffer;
}

/** Base64 without blowing the call stack on long buffers. */
export function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  let bin = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

export class SarvamRealtimeStt {
  private ws: WebSocket | null = null;
  private t0 = 0;
  private readonly cfg: Required<SarvamConfig>;

  constructor(cfg: SarvamConfig) {
    this.cfg = { ...DEFAULTS, ...cfg } as Required<SarvamConfig>;
  }

  private url(): string {
    // Model, encoding and sample rate are the relay's to decide -- it pins them
    // to what the AudioWorklet actually produces. Sending them from here would
    // let a stale client silently contradict the server.
    const q = new URLSearchParams({
      language_code: this.cfg.languageCode,
      stream_type: this.cfg.streamType,
      endpointing: this.cfg.endpointing,
      mode: this.cfg.mode,
      silence_duration_ms: String(this.cfg.silenceDurationMs),
      min_speech_duration_ms: String(this.cfg.minSpeechDurationMs),
    });
    return `${this.cfg.baseUrl}?${q.toString()}`;
  }

  /** Open the socket. Resolves once the server acknowledges the session. */
  async connect(onEvent: (e: TranscriptEvent) => void): Promise<void> {
    this.t0 = performance.now();
    const ws = new WebSocket(this.url());
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(typeof ev.data === "string" ? ev.data : "");
      } catch {
        return;
      }
      const evt = this.decode(msg);
      if (evt) onEvent(evt);
    };

    ws.onerror = () =>
      onEvent({
        kind: "error",
        text: "websocket transport error",
        tMs: this.elapsed(),
        fatal: true,
      });

    ws.onclose = () =>
      onEvent({ kind: "session_end", text: "", tMs: this.elapsed() });

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("stt connect timeout")), 8000);
      ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
    });
  }

  /** Map Sarvam's event envelope onto our provider-neutral shape. */
  private decode(msg: Record<string, unknown>): TranscriptEvent | null {
    const type = String(msg.type ?? msg.event ?? "");
    const tMs = this.elapsed();

    switch (type) {
      case "session.begin":
        return { kind: "session_begin", text: "", tMs, raw: msg };
      case "vad.speech_start":
        return { kind: "speech_start", text: "", tMs, raw: msg };
      case "vad.speech_end":
        return { kind: "speech_end", text: "", tMs, raw: msg };
      case "transcript.partial":
        return { kind: "partial", text: String(msg.transcript ?? msg.text ?? ""), tMs, raw: msg };
      case "transcript.final":
        return { kind: "final", text: String(msg.transcript ?? msg.text ?? ""), tMs, raw: msg };
      case "session.end":
        return { kind: "session_end", text: "", tMs, raw: msg };
      // Emitted by our relay, not by Sarvam: our own server could not reach the
      // vendor. Distinguished in the text so a demo failure is diagnosable
      // without opening the network tab.
      case "relay.error":
        return {
          kind: "error",
          text: `relay: ${String(msg.message ?? msg.code ?? "unreachable")}`,
          tMs,
          fatal: Boolean(msg.is_fatal ?? true),
          raw: msg,
        };
      case "error": {
        const err = (msg.error ?? msg) as Record<string, unknown>;
        return {
          kind: "error",
          text: String(err.message ?? "unknown stt error"),
          tMs,
          fatal: Boolean(err.is_fatal),
          raw: msg,
        };
      }
      default:
        return null;
    }
  }

  /** Send one frame of 16 kHz mono audio. */
  sendAudio(pcm: Float32Array): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(
      JSON.stringify({ event: "audio_input", audio: toBase64(floatTo16BitPcm(pcm)) }),
    );
  }

  /**
   * Tell the server the utterance is over.
   *
   * With client-side VAD we know this a beat before the server would, and that
   * beat is worth more than any model-side optimisation available to us.
   */
  flush(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ event: "flush" }));
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }

  private elapsed(): number {
    return performance.now() - this.t0;
  }
}
