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

import { API_BASE } from "./api";

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
 * client uses so a deployment configures one variable, not two.
 *
 * Two things here are not incidental.
 *
 * **It is called lazily, not at module scope.** This module is imported by a
 * client component that Next still prerenders to HTML in Node at build time, so
 * touching `window` while the module body runs is a build failure, not a
 * runtime one. Resolution happens on `connect()`, in the browser.
 *
 * **The same-origin case is built from `window.location`, not left relative.**
 * When the API serves the frontend, `API_BASE` is `""`; a naive
 * `"".replace(/^http/, "ws")` yields `"/stt/stream"`, and a relative URL in the
 * `WebSocket` constructor is not something to rely on across browsers. The
 * scheme has to be derived anyway — a page served over TLS cannot open an
 * insecure socket — and `location.protocol` is exactly the right source for it.
 */
function defaultRelayUrl(): string {
  if (API_BASE) return `${API_BASE.replace(/^http/, "ws")}/stt/stream`;
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/stt/stream`;
}

const DEFAULTS = {
  token: "",
  languageCode: "en-IN",
  streamType: "fast" as const,
  silenceDurationMs: 300,
  minSpeechDurationMs: 250,
  endpointing: "vad" as const,
  mode: "transcribe" as const,
  //: Resolved on first use rather than here; see `defaultRelayUrl`.
  baseUrl: "",
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
  /** Frames captured before the socket opened. See `sendAudio`. */
  private pending: Float32Array[] = [];
  /** Set when the caller flushed before the socket was ready. */
  private flushRequested = false;
  /** True when at least one frame could not be sent or buffered. */
  private dropped = false;
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
    return `${this.cfg.baseUrl || defaultRelayUrl()}?${q.toString()}`;
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
        // Anything the microphone produced during the handshake is sent now,
        // in order, before the first live frame.
        this.flushPending();
        if (this.flushRequested) {
          this.flushRequested = false;
          ws.send(JSON.stringify({ event: "flush" }));
        }
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
        // Sarvam ends an idle realtime session with
        //   {code: "inactivity_timeout", message: "No audio received for 60s.",
        //    is_fatal: true, status_code: 408}
        // and it means "you stopped talking", not "something broke". The
        // browser VAD deliberately keeps the microphone open after a question,
        // so this fires on any session where the user asks once and then
        // listens to the answer -- which put a red "Speech recognition failed"
        // banner on screen, sometimes directly above a correct answer.
        //
        // Downgraded rather than dropped: `onTranscript` ignores non-fatal
        // errors for display but the event still reaches any caller that wants
        // to observe the session ending.
        const idle =
          String(err.code ?? "") === "inactivity_timeout" ||
          Number(err.status_code ?? 0) === 408;
        return {
          kind: "error",
          text: String(err.message ?? "unknown stt error"),
          tMs,
          fatal: idle ? false : Boolean(err.is_fatal),
          raw: msg,
        };
      }
      default:
        return null;
    }
  }

  /** Send one frame of 16 kHz mono audio. */
  /**
   * Queue one frame of microphone audio.
   *
   * Frames that arrive before the socket is OPEN are **buffered, not dropped**.
   * This used to be `if (readyState !== OPEN) return;`, which silently threw
   * away every frame between the user starting to speak and the WebSocket
   * completing its handshake -- our relay's, then Sarvam's, measured at
   * 300-800 ms. For a two-second question that is the first third of the audio,
   * and it is the worst third to lose: it contains the question word. "What is
   * a corporation?" arrived at the recogniser as "...a corporation?", and short
   * questions could finish and flush before the socket opened at all, so
   * nothing was ever sent and the app simply never answered.
   *
   * The cap is a safety valve, not a tuning knob: it bounds memory if the
   * socket never opens. 16 kHz mono float32 is 64 KB/s, so 400 frames of ~32 ms
   * is roughly 13 seconds of speech.
   */
  sendAudio(pcm: Float32Array): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.flushPending();
      this.ws.send(
        JSON.stringify({ event: "audio_input", audio: toBase64(floatTo16BitPcm(pcm)) }),
      );
      return;
    }
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      if (this.pending.length < 400) this.pending.push(new Float32Array(pcm));
      else this.dropped = true; // buffer full; onSpeechEnd will resend in full
      return;
    }
    // No socket at all yet, or it already closed. Nothing can hold this frame,
    // so record that the stream is incomplete and let `onSpeechEnd` resend the
    // whole utterance rather than transcribe a truncated one.
    this.dropped = true;
  }

  /**
   * Seed the client with audio captured *before* this client existed.
   *
   * The VAD emits `onFrameProcessed` for a frame **before** it emits the
   * `onSpeechStart` that constructs this client -- so the frame carrying the
   * onset of the first word, and every frame of pre-speech padding behind it,
   * is produced while there is nothing yet to send it to. Measured effect:
   * "Can gabapentin treat neuropathy?" reached Sarvam as "And gabapentin treat
   * neuropathy". The question still retrieved correctly, but a demo that
   * visibly mishears its first word reads as a broken microphone.
   *
   * Queued rather than sent, so this is safe to call at any point in the
   * socket's lifecycle: the frames drain in order ahead of the live stream the
   * moment the handshake completes.
   */
  prime(frames: readonly Float32Array[]): void {
    for (const frame of frames) {
      if (this.pending.length >= 400) {
        this.dropped = true;
        break;
      }
      this.pending.push(new Float32Array(frame));
    }
    // A no-op unless the socket is already open, in which case it sends now.
    this.flushPending();
  }

  /**
   * Send the whole utterance, but only if the streaming path dropped any of it.
   *
   * The VAD hands the complete, pre-padded utterance to `onSpeechEnd`. That is
   * strictly more reliable than the frame stream, which races the WebSocket
   * handshake — but sending both would duplicate the audio and therefore the
   * transcript. `dropped` is set only when a frame arrived with no socket at
   * all to buffer it into, which is the one case the buffer cannot rescue.
   */
  sendUtteranceIfIncomplete(audio: Float32Array): void {
    if (!this.dropped || this.ws?.readyState !== WebSocket.OPEN) return;
    this.dropped = false;
    // Chunked because one base64 blob of a long utterance is a large single
    // frame, and Sarvam expects ~100 ms per message.
    const CHUNK = 1600; // 100 ms at 16 kHz
    for (let i = 0; i < audio.length; i += CHUNK) {
      this.ws.send(
        JSON.stringify({
          event: "audio_input",
          audio: toBase64(floatTo16BitPcm(audio.subarray(i, i + CHUNK))),
        }),
      );
    }
  }

  /** Drain frames captured while the socket was still connecting. */
  private flushPending(): void {
    if (!this.pending.length || this.ws?.readyState !== WebSocket.OPEN) return;
    const queued = this.pending;
    this.pending = [];
    for (const frame of queued) {
      this.ws.send(
        JSON.stringify({ event: "audio_input", audio: toBase64(floatTo16BitPcm(frame)) }),
      );
    }
  }

  /**
   * Tell the server the utterance is over.
   *
   * With client-side VAD we know this a beat before the server would, and that
   * beat is worth more than any model-side optimisation available to us.
   */
  flush(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      // A short question can end before the handshake completes. Remember the
      // intent and honour it on open, otherwise the buffered audio sits in the
      // socket with nothing ever telling Sarvam the utterance finished, and the
      // UI waits forever for a final transcript that cannot arrive.
      this.flushRequested = true;
      return;
    }
    this.flushPending();
    this.ws.send(JSON.stringify({ event: "flush" }));
  }

  close(): void {
    this.pending = [];
    this.flushRequested = false;
    this.ws?.close();
    this.ws = null;
  }

  private elapsed(): number {
    return performance.now() - this.t0;
  }
}
