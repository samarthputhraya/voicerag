"use client";

/**
 * VoiceRAG — speak a question, get a grounded answer.
 *
 * Request lifecycle, and why it is shaped this way:
 *
 *   1. Browser VAD detects speech and streams 16 kHz PCM straight to Sarvam.
 *      Audio never touches our server, so the India->US hop is never on the
 *      audio path.
 *   2. Every partial transcript is fired at `/speculate`, which warms the
 *      server's retrieval cache while the user is still talking.
 *   3. VAD detects the endpoint and flushes. This is the moment we start the
 *      clock for the latency we report.
 *   4. The final transcript goes to `/ask/stream`. Retrieval is usually already
 *      cached from step 2, so the server spends its budget on generation.
 *   5. Tokens render as they arrive; the trace lands with the final frame and
 *      the HUD draws the waterfall.
 *
 * The VAD tuning in VAD_OPTS is the single highest-leverage setting in this
 * file — see the comment there.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMicVAD } from "@ricky0123/vad-react";
import type { RealTimeVADOptions } from "@ricky0123/vad-web";

import LatencyHud, { splitBreakdown, type StageBar } from "../components/LatencyHud";
import {
  fetchExamples,
  fetchStats,
  speak,
  speakable,
  speculate,
  streamAnswer,
  type Citation,
  type RagResponse,
  type Stats,
} from "../lib/api";
import { SarvamRealtimeStt, type TranscriptEvent } from "../lib/sarvam";

/**
 * Where the VAD's runtime assets live.
 *
 * `vad-web` defaults `baseAssetPath` and `onnxWASMBasePath` to `"./"`, and that
 * default is the entire reason the microphone never worked in a browser. A
 * relative path is resolved against whatever the *resolver* considers its base,
 * and the three consumers here disagree about what that is:
 *
 *   - `fetch("./silero_vad_legacy.onnx")` and `audioWorklet.addModule("./…")`
 *     resolve against the **document**, so they hit `localhost:3000/…` and
 *     return 200.
 *   - onnxruntime-web loads its WASM glue with `import(url)` carrying a
 *     `webpackIgnore` comment, so it stays a **native** dynamic import — and a
 *     native import resolves against the **importing module**, which under Next
 *     is the bundled chunk in `/_next/static/chunks/app/`. `"./ort-wasm-simd-
 *     threaded.mjs"` therefore resolved to
 *     `/_next/static/chunks/app/ort-wasm-simd-threaded.mjs`, which 404s.
 *
 * That 404 took out the only WASM backend, so `MicVAD.new()` threw
 * "no available backend found", the Silero model never loaded, and every later
 * `vad.start()` became a silent no-op. Verifying that
 * `/ort-wasm-simd-threaded.mjs` returns 200 was misleading — ORT never asked
 * for that URL.
 *
 * Absolute URLs remove the disagreement: every consumer resolves them the same
 * way. Evaluated lazily because this module is also rendered on the server,
 * where `window` does not exist; the root-relative fallback is never the value
 * actually used by the browser.
 */
const ASSET_BASE =
  typeof window === "undefined" ? "/" : `${window.location.origin}/`;

/**
 * ORT's own loader configuration, applied after `vad-web` has set its (string)
 * prefix, so this wins.
 *
 * `wasmPaths` as a string is only a *prefix*, and the prefix is not what breaks
 * — the bundler-rewritten base is. The object form is an explicit per-file
 * override that ORT passes straight to `import()` and `locateFile` without
 * re-resolving, which is the only form that survives Next's chunking.
 *
 * Single-threaded on purpose: the threaded build wants `SharedArrayBuffer`,
 * which needs the COOP/COEP headers Next does not send. ORT detects that and
 * falls back on its own, but only after logging two warnings and re-deciding
 * mid-load; pinning it keeps the model load to one branch.
 */
const configureOrt: NonNullable<RealTimeVADOptions["ortConfig"]> = (ort) => {
  ort.env.logLevel = "error";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = {
    wasm: `${window.location.origin}/ort-wasm-simd-threaded.wasm`,
    mjs: `${window.location.origin}/ort-wasm-simd-threaded.mjs`,
  };
};

/**
 * VAD tuning.
 *
 * The library default for `redemptionMs` is 1400 — it waits 1.4 seconds of
 * silence before deciding you stopped talking. That single default would dwarf
 * every millisecond we save everywhere else in the pipeline combined. At 260ms
 * the endpoint fires almost immediately after a question ends, while still
 * tolerating the natural pause before a final word.
 *
 * `preSpeechPadMs` is kept generous so the first phoneme is never clipped;
 * clipping "what" into "at" costs far more in retrieval quality than 300ms of
 * leading audio costs in latency.
 */
const VAD_OPTS = {
  positiveSpeechThreshold: 0.35,
  negativeSpeechThreshold: 0.25,
  redemptionMs: 260,
  preSpeechPadMs: 300,
  minSpeechMs: 250,
  baseAssetPath: ASSET_BASE,
  onnxWASMBasePath: ASSET_BASE,
  ortConfig: configureOrt,
} as const;

/** How many pre-speech frames to keep. See `preRollRef`. */
const PRE_ROLL_FRAMES = 5;

/**
 * Every language Sarvam's realtime endpoint accepts, which is the whole point of
 * the organisers choosing an ai4bharat corpus. Anything other than `en-IN` is
 * sent with `mode=translate`, so the question arrives at the retriever in
 * English regardless of what was spoken. Measured: translating Hindi speech
 * costs 945 ms against 1006 ms to transcribe it — the multilingual path is not
 * slower, it is marginally faster.
 */
const LANGUAGES = [
  { code: "en-IN", label: "English" },
  { code: "hi-IN", label: "हिन्दी" },
  { code: "bn-IN", label: "বাংলা" },
  { code: "ta-IN", label: "தமிழ்" },
  { code: "te-IN", label: "తెలుగు" },
  { code: "mr-IN", label: "मराठी" },
  { code: "kn-IN", label: "ಕನ್ನಡ" },
  { code: "ml-IN", label: "മലയാളം" },
  { code: "gu-IN", label: "ગુજરાતી" },
  { code: "pa-IN", label: "ਪੰਜਾਬੀ" },
  { code: "or-IN", label: "ଓଡ଼ିଆ" },
  { code: "auto", label: "Auto-detect" },
] as const;

type Phase = "idle" | "listening" | "thinking" | "answering" | "done" | "error";

/**
 * One wording for "the voice model did not load", shared by the effect that
 * notices it and the click handler that refuses to pretend otherwise.
 */
function vadLoadError(errored: string | false): string {
  return (
    `Voice model failed to load: ${
      typeof errored === "string" ? errored : "unknown error"
    }. Typing still works.`
  );
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [language, setLanguage] = useState<string>("en-IN");
  const [partial, setPartial] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [result, setResult] = useState<RagResponse | null>(null);
  const [stages, setStages] = useState<StageBar[]>([]);
  const [pipelineMs, setPipelineMs] = useState(0);
  const [generationMs, setGenerationMs] = useState(0);
  const [examples, setExamples] = useState<string[]>([]);
  const [unanswerable, setUnanswerable] = useState<string | null>(null);
  const [voiceOn, setVoiceOn] = useState(true);
  // Read inside `onFinal`, which captured its closure before the toggle could
  // change; a ref keeps `ask` out of the dependency churn.
  const voiceOnRef = useRef(true);
  const [speaking, setSpeaking] = useState(false);
  // Synthesis is ~1.4 s of Sarvam round trip BEFORE any audio exists. One
  // boolean for both states made the UI claim "Speaking" over silence, which
  // on camera reads as broken audio rather than as latency.
  const [synthesising, setSynthesising] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  /**
   * Generation token for speech. `playAnswer` awaits synthesis for ~1.4 s and
   * then played unconditionally — so muting (or asking again) during that
   * window revoked nothing, and the superseded audio arrived late and spoke
   * anyway, with no control left on screen to stop it. Every interruption
   * bumps the sequence; a synthesis that comes back to a stale sequence
   * discards its audio instead of playing it.
   */
  const speakSeqRef = useRef(0);
  /**
   * The streamed answer, readable from inside the `onFinal` closure.
   * `answer` state is stale there because the closure captured it before the
   * tokens arrived; reading a ref avoids re-creating `ask` on every token.
   */
  const answerRef = useRef("");
  const [history, setHistory] = useState<number[]>([]);
  const [speculations, setSpeculations] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  const sttRef = useRef<SarvamRealtimeStt | null>(null);
  /**
   * The last few frames heard before the VAD called it speech. The legacy
   * Silero model runs on 1536-sample frames, so each is 96 ms and five of them
   * comfortably cover the 300 ms of `preSpeechPadMs` the VAD is configured to
   * prepend, plus the frame that triggered the detection itself.
   */
  const preRollRef = useRef<Float32Array[]>([]);
  const abortRef = useRef<(() => void) | null>(null);
  const lastSpecRef = useRef("");
  const clockRef = useRef(0);

  useEffect(() => {
    fetchExamples(6)
      .then((e) => {
        setExamples(e.examples);
        setUnanswerable(e.unanswerable_example);
      })
      .catch(() => undefined);
    fetchStats().then(setStats).catch(() => setStats(null));
  }, []);

  // --- answer path ----------------------------------------------------------

  /**
   * Stop any answer being spoken — or still being synthesised — and release
   * its blob URL. Bumping the sequence first is what actually cancels an
   * in-flight synthesis; pausing the audio element only handles playback.
   */
  const stopSpeaking = useCallback(() => {
    speakSeqRef.current += 1;
    audioRef.current?.pause();
    audioRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    setSynthesising(false);
    setSpeaking(false);
  }, []);

  /**
   * Synthesise and play one answer.
   *
   * Speaks in the language the user chose, not always English. When they asked
   * in Tamil the retrieval ran in English — that is an implementation detail of
   * the encoder, not something they should hear.
   */
  const playAnswer = useCallback(
    async (text: string) => {
      // Strip citation markers and bracketed notes: they are for the eye.
      // "…treats neuropathy [1][3]" must not be read as "one three".
      const body = speakable(text);
      if (!body) return;
      stopSpeaking();
      const seq = speakSeqRef.current;
      setSynthesising(true);
      const url = await speak(body, language === "auto" ? "en-IN" : language);
      // Superseded while we waited (mute, a new question, another Replay):
      // discard rather than talk over whatever superseded us.
      if (seq !== speakSeqRef.current) {
        if (url) URL.revokeObjectURL(url);
        return;
      }
      setSynthesising(false);
      if (!url) return;
      audioUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => stopSpeaking();
      audio.onerror = () => stopSpeaking();
      try {
        await audio.play();
        setSpeaking(true);
      } catch {
        // Autoplay policy blocks playback until the page has been interacted
        // with. Every path here follows a click or a spoken question, so this
        // is rare; failing quietly beats an error the user cannot act on.
        stopSpeaking();
      }
    },
    [language, stopSpeaking],
  );

  useEffect(() => stopSpeaking, [stopSpeaking]);
  useEffect(() => {
    voiceOnRef.current = voiceOn;
    if (!voiceOn) stopSpeaking();
  }, [voiceOn, stopSpeaking]);

  const ask = useCallback(
    (text: string) => {
      const q = text.trim();
      if (!q) {
        setPhase("idle");
        return;
      }

      abortRef.current?.();
      clockRef.current = performance.now();

      setQuestion(q);
      setAnswer("");
      answerRef.current = "";
      stopSpeaking();
      setCitations([]);
      setResult(null);
      setError(null);
      // Clear the HUD too. Leaving it populated showed the PREVIOUS request's
      // waterfall and "within budget" badge while this one was still
      // retrieving — and left them up forever beside an error banner, a green
      // verdict describing a different question.
      setStages([]);
      setPipelineMs(0);
      setGenerationMs(0);
      setPhase("thinking");

      abortRef.current = streamAnswer(q, language, {
        onToken: (delta) => {
          setPhase("answering");
          answerRef.current += delta;
          setAnswer((a) => a + delta);
        },
        onFinal: (res) => {
          const wall = performance.now() - clockRef.current;
          setResult(res);
          setCitations(res.citations);
          // Unconditional: res.answer is the authoritative text on EVERY exit
          // path. The raw token stream can differ from it — the server strips
          // a leading "ANSWER:" echo and appends a truncation note on a
          // deadline exit — and keeping the stream on screen hid both.
          setAnswer(res.answer);
          // Never sum res.trace.breakdown: it carries deliberately overlapping
          // spans (generate wraps generate.total wraps generate.ttft), and
          // retrieve.dense/retrieve.sparse run concurrently. splitBreakdown
          // keeps the leaves, orders them chronologically rather than by size,
          // and separates the hosted-LLM leg from the path we claim against.
          const split = splitBreakdown(res.trace.breakdown);
          setStages(split.pipeline);
          setPipelineMs(split.pipelineMs);
          setGenerationMs(split.generationMs);
          setHistory((h) => [...h, split.pipelineMs]);
          setPhase("done");
          if (process.env.NODE_ENV === "development") {
            console.debug("wall-clock incl. network", wall.toFixed(1), "ms");
          }
          // Speak it. Deliberately after the text has rendered: synthesis costs
          // ~1.4 s, and blocking the answer on it would spend the time-to-first-
          // token we optimised hardest for. The refusal text is spoken too — a
          // system that only talks when it succeeds teaches you to distrust its
          // silence. playAnswer strips the citation markers before Sarvam
          // sees them.
          if (voiceOnRef.current && res.answer.trim()) void playAnswer(res.answer);
        },
        onError: (e) => {
          setError(e.message);
          setPhase("error");
        },
      });
    },
    [language],
  );

  // --- speech path ----------------------------------------------------------

  const onTranscript = useCallback(
    (e: TranscriptEvent) => {
      switch (e.kind) {
        case "partial":
          setPartial(e.text);
          // Warm the retrieval cache while the user is still speaking.
          if (e.text !== lastSpecRef.current) {
            lastSpecRef.current = e.text;
            speculate(e.text, language);
            setSpeculations((n) => n + 1);
          }
          break;
        case "final":
          setPartial("");
          sttRef.current?.close();
          sttRef.current = null;
          ask(e.text);
          break;
        case "error":
          if (e.fatal) {
            setError(`Speech recognition failed: ${e.text}`);
            setPhase("error");
            sttRef.current?.close();
            sttRef.current = null;
          }
          break;
        default:
          break;
      }
    },
    [ask, language],
  );

  const vad = useMicVAD({
    ...VAD_OPTS,
    startOnLoad: false,
    onSpeechStart: () => {
      setPhase("listening");
      setPartial("");
      setSpeculations(0);
      lastSpecRef.current = "";

      const stt = new SarvamRealtimeStt({
        languageCode: language,
        // Ask Sarvam to translate rather than transcribe whenever the user
        // is not already speaking English. The corpus is indexed in English
        // and the query encoder cannot represent Indic scripts, so the
        // transcript is the right place to cross the language boundary —
        // and it is free, because Sarvam does it in the same call.
        mode: language === "en-IN" ? "transcribe" : "translate",
      });
      // Hand over the audio the VAD had already heard before it decided this
      // was speech. Without it the recogniser never receives the onset of the
      // first word, because the VAD reports a frame to `onFrameProcessed`
      // before it reports the speech start that creates this client.
      stt.prime(preRollRef.current);
      preRollRef.current = [];

      // Publish the client BEFORE awaiting the handshake. `onFrameProcessed`
      // fires every ~96 ms from this moment, and until this ref is set those
      // frames go nowhere. Assigning first lets them land in the client's
      // pending buffer, which drains in order the instant the socket opens.
      // Assigning after `await` was the bug that made short questions produce
      // no answer at all.
      sttRef.current = stt;

      void stt.connect(onTranscript).catch((e: unknown) => {
        setError(`Speech recognition failed: ${(e as Error).message}`);
        setPhase("error");
        sttRef.current?.close();
        sttRef.current = null;
      });
    },
    onFrameProcessed: (_p, frame: Float32Array) => {
      const stt = sttRef.current;
      if (stt) {
        stt.sendAudio(frame);
        return;
      }
      // No client yet, so this frame is either silence or -- for the last few
      // frames before the VAD makes up its mind -- the beginning of the
      // question. Keep a short rolling window of them rather than discarding
      // the word the user actually opened with. `onSpeechStart` claims it.
      const roll = preRollRef.current;
      roll.push(new Float32Array(frame));
      if (roll.length > PRE_ROLL_FRAMES) roll.shift();
    },
    onSpeechEnd: (audio: Float32Array) => {
      // The VAD hands back the COMPLETE utterance here, pre-speech padding
      // included. Streaming frames is what gives us partial transcripts (and
      // therefore speculative retrieval), but it is also the fragile path: it
      // races the WebSocket handshake, and any frame lost before the socket
      // opened is lost for good.
      //
      // So this is belt and braces. If the socket was not open in time to
      // receive the whole utterance frame-by-frame, send it whole now. The
      // relay is a byte pump, so a second copy of already-sent audio would
      // duplicate the transcript -- hence `sendUtteranceIfIncomplete`, which
      // sends only when the streaming path is known to have missed frames.
      sttRef.current?.sendUtteranceIfIncomplete(audio);
      sttRef.current?.flush();
      setPhase("thinking");
    },
  });

  // The VAD reports load failures here -- a 404 on the ONNX weights, a blocked
  // AudioWorklet, an insecure origin. Without surfacing it the button simply
  // sits on "Loading voice model…" forever and the microphone silently never
  // works, which is indistinguishable from the app ignoring you.
  useEffect(() => {
    if (vad.errored) {
      setError(vadLoadError(vad.errored));
      setPhase("error");
    }
  }, [vad.errored]);

  const toggleMic = useCallback(() => {
    if (vad.listening) {
      vad.pause();
      sttRef.current?.close();
      sttRef.current = null;
      // Otherwise up to half a second of pre-pause room tone would be
      // prepended to whatever the user says after starting the mic again.
      preRollRef.current = [];
      setPhase("idle");
      return;
    }
    // `useMicVAD`'s `start()` is a no-op when the model failed to load: it is
    // written as `if (!loading && !errored) { await vad?.start() }` and then
    // resolves either way. So an errored VAD produced a *successful* promise,
    // this handler cleared the error banner and set the phase to "listening",
    // and the UI sat there claiming to hear you with no microphone open and
    // nothing on screen to explain it. That is the state the demo was stuck in.
    // Check first, and keep the diagnosis on screen instead of erasing it.
    if (vad.errored) {
      setError(vadLoadError(vad.errored));
      setPhase("error");
      return;
    }
    setError(null);
    // Rejects when the user denies the microphone or the page is not on a
    // secure origin. Firing it unawaited and unconditionally claiming
    // "listening" produced the same lie for a different reason.
    // Ask by voice OR by typing -- both reach the same pipeline.
    void Promise.resolve(vad.start())
      .then(() => setPhase("listening"))
      .catch((e: unknown) => {
        setError(
          `Microphone unavailable: ${(e as Error)?.message ?? "permission denied"}. ` +
            `You can still type your question below.`,
        );
        setPhase("error");
      });
  }, [vad]);

  const [typed, setTyped] = useState("");

  const submitTyped = useCallback(
    (text: string) => {
      const q = text.trim();
      if (!q) return;
      setTyped("");
      // Not setPartial: that renders under "hearing", which is a lie for a
      // typed question and made the UI look like it had misheard something
      // nobody said. `ask` sets the "asked" line itself.
      setPartial("");
      setPhase("thinking");
      void ask(q);
    },
    [ask],
  );

  useEffect(() => () => {
    abortRef.current?.();
    sttRef.current?.close();
  }, []);

  // --- render ---------------------------------------------------------------

  const ttft = result?.trace.breakdown["generate.ttft"] ?? null;
  const abstained = result?.abstained ?? false;

  /**
   * Citation labels, re-derived from the answer text.
   *
   * The server returns citations in FIRST-APPEARANCE order of the [n] markers,
   * but carries no original index — so labelling boxes by array position made
   * an answer reading "…sedation [3][1]." point at Sources labelled [1] and
   * [2]: the marker on screen named a different box. Walking the answer's own
   * markers in order reproduces the server's ordering exactly. The length
   * guard falls back to positional labels on the uncited path, where the
   * server returns the whole retrieved context and positional is correct.
   */
  const citeLabels = useMemo(() => {
    const seen: string[] = [];
    for (const m of answer.matchAll(/\[(\d{1,2})\]/g)) {
      if (!seen.includes(m[1])) seen.push(m[1]);
    }
    return seen.length === citations.length
      ? seen
      : citations.map((_, i) => String(i + 1));
  }, [answer, citations]);

  /** Did the answer actually cite, or is the panel retrieved-context fallback? */
  const citedDirectly =
    citations.length > 0 && /\[\d{1,2}\]/.test(answer);

  /**
   * Citations as they should be READ, not as they are stored.
   *
   * The server sends the raw chunk (700 chars, 120 of which overlap the
   * neighbouring chunk), and MS MARCO passages carry scraped nav boilerplate:
   * alphabet runs ("a b c d e f g…"), bare URLs, and question-list fragments.
   * Cleaning is display-only — scores, retrieval and the prompt are untouched
   * — and the full original text stays available on the title attribute.
   */
  const displayCitations = useMemo(() => {
    const seenStart = new Set<string>();
    const out: { c: Citation; label: string; text: string }[] = [];
    citations.forEach((c, i) => {
      // Scraped furniture (A-Z index strips, bare URLs, question rails) is
      // already gone: the server cleans Citation.text in _citations, so there
      // is one implementation of those rules rather than two that drift.
      // What remains here is purely a display decision.
      let t = c.text;
      // Adjacent chunks of one document overlap by 120 characters, so two
      // citations from the same page open identically and read as a
      // copy-paste bug. Only the uncited context list may drop one — a cited
      // box has to exist for its [n] marker to point at.
      const key = t.slice(0, 80).toLowerCase();
      if (!citedDirectly && seenStart.has(key)) return;
      seenStart.add(key);
      // Enough to verify a claim against. Five unclipped 700-char slabs pushed
      // the guardrails and the latency HUD below the fold on camera.
      if (t.length > 240) {
        const cut = t.lastIndexOf(" ", 240);
        t = `${t.slice(0, cut > 160 ? cut : 240)}…`;
      }
      out.push({ c, label: citeLabels[i] ?? String(i + 1), text: t });
    });
    return out;
  }, [citations, citeLabels, citedDirectly]);

  return (
    <main>
      <header className="top">
        <div>
          <div className="brand-mark">
            <svg className="wave" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M1 15c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
              />
              <path
                d="M1 20c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
                opacity="0.45"
              />
              <circle cx="12" cy="6" r="3.1" stroke="currentColor" strokeWidth="1.9" />
            </svg>
            <span className="eyebrow">HH Goa 2026 · Task 2 · Voice RAG</span>
          </div>
          <h1>
            Less noise.
            <br />
            <span className="thin">More signal.</span>
          </h1>
          <p className="sub">
            Ask out loud in eleven Indian languages. Every answer is grounded in
            retrieved passages, cited, and comes with a receipt for every
            millisecond it cost.
          </p>
        </div>
        <div className="top-right">
          <span className="eyebrow">Spoken language</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            aria-label="Spoken language"
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      <section className="ask">
        <div className="mic-row">
          <button
            className={`mic ${vad.listening ? "on" : ""} ${phase}`}
            onClick={toggleMic}
            disabled={vad.loading}
            aria-pressed={vad.listening}
          >
            {/*
              The organisers draw exactly this on their own Task #2 card: a
              green disc inside a dashed pink ring, with a small yellow badge
              clipped to the corner. There it is an illustration of a
              microphone; here it is the microphone. The badge keeps the job it
              already looks like it is doing — it turns pink and pulses while
              the socket is open, and the ring turns.
            */}
            <span className="mic-emblem" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <rect
                  x="9"
                  y="2"
                  width="6"
                  height="11"
                  rx="3"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <path
                  d="M5 10.5a7 7 0 0 0 14 0"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <path
                  d="M12 17.5V21"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <span className="mic-label">
              {vad.loading
                ? "Loading voice model…"
                : vad.listening
                  ? "Listening — click to stop"
                  : "Hold a question. Click to speak."}
            </span>
          </button>

          <PhasePill phase={phase} speculations={speculations} />
        </div>

        {/*
          The typed path is not a fallback bolted on for accessibility -- it is
          the only way this system is demonstrable when the microphone is
          unavailable, which on a judge's managed browser or a phone with a
          reflexive "Block" tap is a coin flip. Everything downstream of the
          transcript is identical, so a typed question exercises retrieval,
          guardrails, grounding and the HUD exactly as a spoken one does.
        */}
        <form
          className="type-row"
          onSubmit={(e) => {
            e.preventDefault();
            submitTyped(typed);
          }}
        >
          {/*
            readOnly, not disabled: disabling a focused element dumps keyboard
            focus to <body>, so every typed question stranded a keyboard user
            back at the top of the page — and the typed path exists precisely
            for the judge on a managed browser.
          */}
          <input
            type="text"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="…or type a question"
            aria-label="Type a question"
            readOnly={phase === "thinking" || phase === "answering"}
            aria-busy={phase === "thinking" || phase === "answering"}
          />
          <button type="submit" disabled={!typed.trim()}>
            Ask
          </button>
        </form>

        {/*
          Sampled from the shard's own labelled queries, not invented. The most
          common way this demo looks broken is a fair question the corpus was
          never going to answer -- it indexes a fixed slice of MS MARCO, not the
          web -- and the system then correctly declines, which from the outside
          is indistinguishable from a bug.
        */}
        {(examples.length > 0 || unanswerable) && (
          <>
            <p className="chips-head eyebrow">
              {examples.length > 0
                ? "This corpus can answer questions like"
                : "Watch it decline a question it cannot answer"}
            </p>
            <div className="chips">
              {examples.map((p) => (
                <button key={p} className="chip" onClick={() => submitTyped(p)}>
                  {p}
                </button>
              ))}
              {/*
                Outside the examples guard on purpose: the unanswerable chip
                comes from a separate manifest key and is the single best demo
                affordance on the page — it shows the abstention guardrail
                working on purpose. It must not vanish just because the
                sampled example pool came back empty.
              */}
              {unanswerable && (
                <button
                  className="chip decline"
                  onClick={() => submitTyped(unanswerable)}
                  title="Labelled unanswerable in MS MARCO — watch it decline rather than guess"
                >
                  {unanswerable}
                </button>
              )}
            </div>
          </>
        )}
      </section>

      {partial && (
        <p className="partial" aria-live="polite">
          <span className="pl">hearing</span> {partial}
          <span className="caret" />
        </p>
      )}

      {question && (
        <p className="question">
          <span className="pl">asked</span> {question}
        </p>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {/*
        Keyed on the request state, not on the text. Keying on `answer` did two
        bad things: during "thinking" the card did not exist at all, so a
        hosted-LLM round trip was seconds of motionless screen that read as a
        hang on camera; and an abstention whose reason string was empty would
        have removed the card entirely, taking the "Declined to answer" badge
        and the mute control with it.
      */}
      {(phase === "thinking" || phase === "answering" || result) && (
        <article className={`answer ${abstained ? "abstained" : ""}`}>
          {abstained && <div className="abstain-tag">Declined to answer</div>}
          <p aria-live="polite">
            {answer ||
              (phase === "thinking" ? (
                <span className="searching">
                  searching{" "}
                  {stats ? stats.chunks.toLocaleString() : "the"} indexed
                  passages
                </span>
              ) : abstained ? (
                "The system declined to answer."
              ) : null)}
            {(phase === "answering" || phase === "thinking") && (
              <span className="caret" />
            )}
          </p>
          <div className="mic-row" style={{ marginTop: 12, gap: 16 }}>
            {(speaking || synthesising) && (
              <span
                className={`speaking-tag ${synthesising ? "synthesising" : ""}`}
              >
                <span className="bars" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
                {synthesising ? "Preparing voice…" : "Speaking"}
              </span>
            )}
            <button
              className="mute-btn"
              onClick={() => setVoiceOn((v) => !v)}
              aria-pressed={!voiceOn}
            >
              {voiceOn ? "Mute spoken answers" : "Unmute spoken answers"}
            </button>
            {!speaking && !synthesising && voiceOn && answer.trim() && phase === "done" && (
              <button className="mute-btn" onClick={() => void playAnswer(answer)}>
                Replay
              </button>
            )}
          </div>
        </article>
      )}

      {displayCitations.length > 0 && (
        <section className="cites">
          {/*
            "Sources" is only honest when the answer actually cited. When the
            model emits no [n], the server falls back to returning the whole
            retrieved context — presenting that under "Sources" with pink
            markers claimed a link the answer never made.
          */}
          <h2>{citedDirectly ? "Sources" : "Retrieved passages"}</h2>
          {!citedDirectly && (
            <p className="cites-note">
              shown as context — the answer did not cite these directly
            </p>
          )}
          <ol role="list">
            {displayCitations.map(({ c, label, text }) => (
              <li key={c.chunk_id} role="listitem" title={c.text}>
                <span className="ci">[{label}]</span>
                <span className="ct">{text}</span>
                <span className="cs">{c.score.toFixed(3)}</span>
              </li>
            ))}
          </ol>
        </section>
      )}

      {result?.guardrails && <GuardrailPanel report={result.guardrails} />}

      {/*
        200 ms is the *claim*, not the serving deadline. The served budget is
        2500 ms because a real hosted LLM call costs 450-900 ms of round trip;
        scoring the HUD against that would make the bar meaningless. The bar
        here measures the retrieval path, which is what the brief scopes and
        what this system actually engineers.
      */}
      <LatencyHud
        stages={stages}
        history={history}
        ttftMs={ttft}
        budgetMs={200}
        pipelineMs={pipelineMs}
        generationMs={generationMs}
      />

      {stats && (
        <footer className="foot">
          {stats.chunks.toLocaleString()} chunks · {stats.strategy} chunking ·{" "}
          {stats.embedding_model}
          {/* provider is the literal string "none" on every refusal and every
              provider failure — "answered by none" at the exact moment the
              demo is degraded is the worst possible caption. */}
          {result && result.provider !== "none" && ` · answered by ${result.provider}`}
        </footer>
      )}
    </main>
  );
}

function PhasePill({ phase, speculations }: { phase: Phase; speculations: number }) {
  const label: Record<Phase, string> = {
    idle: "Ready",
    listening: "Listening",
    thinking: "Retrieving",
    answering: "Answering",
    done: "Done",
    error: "Error",
  };
  return (
    <div className={`pill ${phase}`}>
      {label[phase]}
      {speculations > 0 && phase !== "idle" && (
        <span className="spec" title="Retrievals speculatively issued on partial transcripts">
          {speculations} speculative {speculations === 1 ? "search" : "searches"}
        </span>
      )}
    </div>
  );
}

function GuardrailPanel({
  report,
}: {
  report: NonNullable<RagResponse["guardrails"]>;
}) {
  // Grounding only means something once an answer exists to ground. Showing
  // "Grounding 0%" beside a refusal reads as a failed check when in fact
  // nothing was ever asserted, which is the guardrail succeeding.
  //
  // typeof, not !== undefined: the API serialises "grounding did not run" as
  // JSON null, and `null !== undefined` is true — so the guard never fired and
  // a skipped check rendered as a green "0%" captioned "every claim traced".
  const scored =
    !report.abstained && typeof report.grounding_score === "number";
  const grounding = scored
    ? `${((report.grounding_score ?? 0) * 100).toFixed(0)}%`
    : "not applicable";

  // The gate's own confidence that the corpus cannot answer — the one number
  // the answer card cannot show.
  const gatePct = ((report.abstain_confidence ?? 0) * 100).toFixed(0);

  return (
    <section className="guard">
      <h2>Guardrails</h2>
      <div className="guard-grid">
        <GuardCard
          k="Input"
          // input_reason is None on every ALLOWED request, so the happy path —
          // the whole demo — rendered a half-empty box. State what the check
          // looked for instead of leaving dead space.
          v={report.input_allowed ? "Accepted" : "Blocked"}
          note={
            report.input_allowed
              ? "no injection, jailbreak or policy pattern matched"
              : report.input_reason
          }
          state={report.input_allowed ? "pass" : "fail"}
        />
        <GuardCard
          k="Answer"
          v={report.abstained ? "Declined" : "Asserted"}
          // Never abstain_reason: on every abstained exit the response answer
          // IS abstain_reason, so this card was reprinting — verbatim — the
          // refusal sentence the judge just read in the answer card above it.
          // The gate confidence is the number that card cannot show.
          note={
            report.abstained
              ? `abstention gate ${gatePct}% confident the corpus cannot answer this`
              : `gate confidence ${gatePct}% — under the refusal bar, so answering was allowed`
          }
          // Declining is the system working, not failing. Colouring it red
          // would teach a viewer to read a correct refusal as a bug.
          state={report.abstained ? "neutral" : "pass"}
        />
        <GuardCard
          k="Grounding"
          v={grounding}
          // The score is a length-weighted mean of lexical support per claim,
          // not a coverage count — captioning 71% as "every claim traced"
          // put two contradictory statements in the one panel whose whole job
          // is credibility. Describe the metric; let the number be the claim.
          note={
            report.unsupported_claims?.length
              ? `${report.unsupported_claims.length} unsupported claim(s) withheld`
              : scored
                ? "mean lexical support per claim, against the cited passages"
                : "no claim was asserted, so there is nothing to ground"
          }
          state={!scored ? "neutral" : report.grounded === false ? "fail" : "pass"}
        />
      </div>
    </section>
  );
}

function GuardCard({
  k,
  v,
  note,
  state,
}: {
  k: string;
  v: string;
  note?: string | null;
  state: "pass" | "fail" | "neutral";
}) {
  return (
    <div className={`guard-card ${state}`}>
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {note && <div className="n">{note}</div>}
    </div>
  );
}
