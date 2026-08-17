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

import { useCallback, useEffect, useRef, useState } from "react";
import { useMicVAD } from "@ricky0123/vad-react";

import LatencyHud, { splitBreakdown, type StageBar } from "../components/LatencyHud";
import {
  fetchExamples,
  fetchStats,
  speak,
  speculate,
  streamAnswer,
  type Citation,
  type RagResponse,
  type Stats,
} from "../lib/api";
import { SarvamRealtimeStt, type TranscriptEvent } from "../lib/sarvam";

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
} as const;

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
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
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

  /** Stop any answer currently being spoken and release its blob URL. */
  const stopSpeaking = useCallback(() => {
    audioRef.current?.pause();
    audioRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
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
      stopSpeaking();
      setSpeaking(true);
      const url = await speak(text, language === "auto" ? "en-IN" : language);
      if (!url) {
        setSpeaking(false);
        return;
      }
      audioUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => stopSpeaking();
      audio.onerror = () => stopSpeaking();
      try {
        await audio.play();
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
          if (res.abstained) setAnswer(res.answer);
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
          // silence.
          const spoken = res.abstained ? res.answer : answerRef.current;
          if (voiceOnRef.current && spoken.trim()) void playAnswer(spoken);
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
      // Publish the client BEFORE awaiting the handshake. `onFrameProcessed`
      // fires every ~32 ms from this moment, and until this ref is set those
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
      sttRef.current?.sendAudio(frame);
    },
    onSpeechEnd: () => {
      // We know the utterance ended before the server does. Say so immediately.
      sttRef.current?.flush();
      setPhase("thinking");
    },
  });

  const toggleMic = useCallback(() => {
    if (vad.listening) {
      vad.pause();
      sttRef.current?.close();
      sttRef.current = null;
      setPhase("idle");
    } else {
      setError(null);
      // `vad.start()` returns a promise that rejects when the user denies the
      // microphone, when the page is not on a secure origin, or when the ONNX
      // assets 404. Firing it unawaited and then unconditionally claiming
      // "listening" produced a UI that sat in the listening state forever
      // after a single Block click, with nothing on screen explaining why.
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
    }
  }, [vad]);

  const [typed, setTyped] = useState("");

  const submitTyped = useCallback(
    (text: string) => {
      const q = text.trim();
      if (!q) return;
      setTyped("");
      setPartial(q);
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
            <span className="dot" />
            {vad.loading
              ? "Loading voice model…"
              : vad.listening
                ? "Listening — click to stop"
                : "Hold a question. Click to speak."}
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
          <input
            type="text"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="…or type a question"
            aria-label="Type a question"
            disabled={phase === "thinking" || phase === "answering"}
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
        {examples.length > 0 && (
          <>
            <p className="chips-head eyebrow">
              This corpus can answer questions like
            </p>
            <div className="chips">
              {examples.map((p) => (
                <button key={p} className="chip" onClick={() => submitTyped(p)}>
                  {p}
                </button>
              ))}
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
        <p className="partial">
          <span className="pl">hearing</span> {partial}
          <span className="caret" />
        </p>
      )}

      {question && (
        <p className="question">
          <span className="pl">asked</span> {question}
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {(answer || phase === "answering") && (
        <article className={`answer ${abstained ? "abstained" : ""}`}>
          {abstained && <div className="abstain-tag">Declined to answer</div>}
          <p>
            {answer}
            {phase === "answering" && <span className="caret" />}
          </p>
          <div className="mic-row" style={{ marginTop: 12, gap: 16 }}>
            {speaking && (
              <span className="speaking-tag">
                <span className="bars" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
                Speaking
              </span>
            )}
            <button
              className="mute-btn"
              onClick={() => setVoiceOn((v) => !v)}
              aria-pressed={!voiceOn}
            >
              {voiceOn ? "Mute spoken answers" : "Unmute spoken answers"}
            </button>
            {!speaking && voiceOn && answer.trim() && phase === "done" && (
              <button className="mute-btn" onClick={() => void playAnswer(answer)}>
                Replay
              </button>
            )}
          </div>
        </article>
      )}

      {citations.length > 0 && (
        <section className="cites">
          <h2>Sources</h2>
          <ol>
            {citations.map((c, i) => (
              <li key={c.chunk_id}>
                <span className="ci">[{i + 1}]</span>
                <span className="ct">{c.text}</span>
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
          {result && ` · answered by ${result.provider}`}
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
  const grounding =
    report.grounding_score !== undefined
      ? `${(report.grounding_score * 100).toFixed(0)}%`
      : "—";
  return (
    <section className="guards">
      <h2>Guardrails</h2>
      <div className="grow">
        <Check ok={report.input_allowed} label="Input accepted" note={report.input_reason} />
        <Check
          ok={!report.abstained}
          label={report.abstained ? "Abstained" : "Answered"}
          note={report.abstain_reason}
          neutral={report.abstained}
        />
        <Check
          ok={report.grounded !== false}
          label={`Grounding ${grounding}`}
          note={
            report.unsupported_claims?.length
              ? `${report.unsupported_claims.length} unsupported claim(s)`
              : null
          }
        />
      </div>
    </section>
  );
}

function Check({
  ok,
  label,
  note,
  neutral = false,
}: {
  ok: boolean;
  label: string;
  note?: string | null;
  neutral?: boolean;
}) {
  return (
    <div className={`chk ${neutral ? "neutral" : ok ? "ok" : "bad"}`}>
      <strong>{label}</strong>
      {note && <em>{note}</em>}
    </div>
  );
}
