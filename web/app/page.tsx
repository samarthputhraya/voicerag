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

import LatencyHud, { type StageBar } from "../components/LatencyHud";
import {
  fetchStats,
  mintSttToken,
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

const LANGUAGES = [
  { code: "en-IN", label: "English" },
  { code: "hi-IN", label: "हिन्दी" },
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
  const [history, setHistory] = useState<number[]>([]);
  const [speculations, setSpeculations] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  const sttRef = useRef<SarvamRealtimeStt | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const lastSpecRef = useRef("");
  const clockRef = useRef(0);

  useEffect(() => {
    fetchStats().then(setStats).catch(() => setStats(null));
  }, []);

  // --- answer path ----------------------------------------------------------

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
      setCitations([]);
      setResult(null);
      setError(null);
      setPhase("thinking");

      abortRef.current = streamAnswer(q, language, {
        onToken: (delta) => {
          setPhase("answering");
          setAnswer((a) => a + delta);
        },
        onFinal: (res) => {
          const wall = performance.now() - clockRef.current;
          setResult(res);
          setCitations(res.citations);
          if (res.abstained) setAnswer(res.answer);
          setStages(
            Object.entries(res.trace.breakdown)
              .map(([name, ms]) => ({ name, ms }))
              .sort((a, b) => b.ms - a.ms),
          );
          setHistory((h) => [...h, res.trace.total_ms]);
          setPhase("done");
          if (process.env.NODE_ENV === "development") {
            console.debug("wall-clock incl. network", wall.toFixed(1), "ms");
          }
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

      void (async () => {
        try {
          const { token } = await mintSttToken();
          const stt = new SarvamRealtimeStt({ token, languageCode: language });
          await stt.connect(onTranscript);
          sttRef.current = stt;
        } catch (e) {
          setError((e as Error).message);
          setPhase("error");
        }
      })();
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
      vad.start();
      setPhase("listening");
    }
  }, [vad]);

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
          <h1>VoiceRAG</h1>
          <p className="sub">
            Speak a question — get an answer grounded in MS MARCO, with citations
            and a receipt for every millisecond.
          </p>
        </div>
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
      </header>

      <section className="mic-row">
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
              : "Click to speak"}
        </button>

        <PhasePill phase={phase} speculations={speculations} />
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

      <LatencyHud stages={stages} history={history} ttftMs={ttft} budgetMs={200} />

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
