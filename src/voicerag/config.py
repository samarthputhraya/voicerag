"""Every tunable in the system, in one place, loaded from the environment.

Three rules govern this module.

**One source of truth.** The API server, the pipeline, the guardrails and the
benchmark all read their numbers from :class:`Settings`. A threshold that lives
in two places eventually differs in two places, and then the report and the demo
disagree about what was measured.

**The latency budget is explicit and per-stage, not a single number.** The
organisers' bar -- "chunking + vector DB retrieval + everything through to final
output ... under 200ms" -- is a *total*. A total alone is unenforceable at
runtime: by the time you notice you have blown it, you have already blown it. So
the budget is decomposed into named stage allowances that sum to less than the
total (:attr:`Settings.budget_headroom_ms` is the slack), and each stage receives
its own slice of the shared
:class:`~voicerag.harness.resilience.Deadline`. A stage that overruns is visible
in the trace as the stage that overran.

**Secrets are ``SecretStr`` and never logged.** :meth:`Settings.public_dict`
exists precisely so that the ``/healthz`` and ``/stats`` handlers cannot leak a
key by accident: it returns presence booleans, not values.

Environment variable names are the field names upper-cased, with no prefix
(``GROQ_API_KEY``, ``INDEX_DIR``, ``BUDGET_GENERATE_MS``), read from the process
environment and from a ``.env`` file if one exists. See ``.env.example`` for the
documented list.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings", "REPO_ROOT"]

#: Repository root, derived from this file's location (``src/voicerag/config.py``).
#: Used to resolve relative paths so that ``uvicorn`` started from any working
#: directory finds the same index.
REPO_ROOT = Path(__file__).resolve().parents[2]

FusionName = Literal["rrf", "minmax", "zscore", "dense", "sparse"]


class Settings(BaseSettings):
    """Runtime configuration for the whole service.

    Instantiate through :func:`get_settings` so the ``.env`` file is parsed once
    per process; construct directly (``Settings(index_dir=...)``) in tests, where
    an isolated instance is exactly what you want.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- deployment -----------------------------------------------------------

    app_name: str = Field(default="voicerag", description="Service name in logs.")
    environment: Literal["dev", "test", "prod"] = Field(
        default="dev", description="Deployment environment; affects log verbosity only."
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_queries: bool = Field(
        default=False,
        description=(
            "Log transcript text. Off by default: transcripts are user speech, "
            "and a latency log is not a place to accumulate personal data."
        ),
    )
    host: str = "0.0.0.0"
    port: int = 8000
    # ``NoDecode`` stops pydantic-settings from JSON-parsing the raw env value,
    # so ``CORS_ORIGINS=http://a,http://b`` works. Without it the only accepted
    # form is a JSON array, which is a hostile thing to put in a shell.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description=(
            "Allowed browser origins, comma-separated or as a JSON array. The "
            "demo frontend runs on :3000. Irrelevant when STATIC_DIR is set and "
            "the frontend is served from this origin -- which is the point of "
            "serving it from this origin."
        ),
    )
    static_dir: Path | None = Field(
        default=None,
        description=(
            "Directory of a built frontend (`web/out`, from `next build` with "
            "output: 'export'). When set and present, it is mounted at / and "
            "the endpoint map moves to /api; when unset the API serves JSON at "
            "/ exactly as before, which is what local development wants because "
            "`next dev` is already serving the frontend on :3000.\n\n"
            "Same-origin is not a convenience. CORS does not apply to WebSocket "
            "upgrades, so `stt_relay._origin_allowed` is the only thing gating "
            "the relay, and a cross-origin deployment that forgets to name its "
            "frontend in CORS_ORIGINS gets a microphone button that lights up, "
            "a waveform that moves, and no transcript that ever arrives. Served "
            "from one origin there is nothing to forget."
        ),
    )

    # -- provider credentials -------------------------------------------------

    sarvam_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    stt_signing_key: SecretStr | None = Field(
        default=None,
        description=(
            "HMAC key used to sign browser STT capabilities when no vendor "
            "ephemeral-credential endpoint is configured. A random per-process "
            "key is generated when unset, which is safe but invalidates issued "
            "tokens on restart."
        ),
    )

    # -- models ---------------------------------------------------------------

    groq_model: str = "openai/gpt-oss-20b"
    groq_reasoning_effort: str | None = "low"
    gemini_model: str = "gemini-2.5-flash-lite"
    openai_model: str = "gpt-4o-mini"
    sarvam_model: str = "saaras:v3-realtime"
    #: Text-to-speech, so the answer is spoken and not merely printed. The brief
    #: asks for voice in *and* out; without this the second half stops at text.
    sarvam_tts_model: str = "bulbul:v2"
    sarvam_tts_speaker: str = "anushka"
    enable_tts: bool = True
    sarvam_batch_model: str = "saaras:v3"
    elevenlabs_model: str = "scribe_v2_realtime"
    stt_language: str = Field(
        default="en-IN", description="BCP-47 code, or 'auto' for language ID."
    )
    stt_stream_type: Literal["fast", "balanced", "simulated"] = "fast"
    sarvam_token_url: str | None = Field(
        default=None,
        description=(
            "Vendor endpoint that mints an ephemeral browser credential. When "
            "set, POST /stt/token proxies to it. When unset the server issues a "
            "signed capability instead -- see voicerag.api.stt_token."
        ),
    )
    stt_token_ttl_s: int = Field(default=300, ge=30, le=3600)

    # -- generation -----------------------------------------------------------

    max_tokens: int = Field(
        default=80,
        ge=1,
        le=512,
        description=(
            "Output-token ceiling. At Groq throughput one token costs ~1 ms, so "
            "this is a latency parameter first and a style parameter second."
        ),
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # -- index ----------------------------------------------------------------

    index_dir: Path = Field(
        default=REPO_ROOT / "data" / "index",
        description="Directory written by scripts/ingest.py.",
    )
    embedder_spec: str = Field(
        default="static:minishlab/potion-base-8M",
        description=(
            "Fallback embedder spec when the index manifest does not name one. "
            "The manifest wins whenever an index is loaded. Measured on this "
            "corpus: static:minishlab/potion-base-8M encodes a query in 0.2 ms "
            "P50, against 20.3 ms P50 / 53.2 ms P100 for lsa:256 -- roughly "
            "320x, and the LSA tail alone pushed retrieval past its slice of "
            "the budget. Use 'lsa:<dim>' only where no download is possible."
        ),
    )
    #: Only consulted when the manifest omits it. `recursive` because the
    #: ablation says so: R@10 0.9070 against 0.8023 for sentence_window, at a
    #: third of the chunk count. See reports/ablation.md.
    chunking_strategy: str = "recursive"
    retrieval_k: int = Field(default=5, ge=1, le=50)
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    ef_search: int = Field(default=64, ge=1, le=512)
    fusion: FusionName = "rrf"
    mmap_sparse: bool = True

    # -- latency budget, per stage --------------------------------------------

    budget_total_ms: float = Field(
        default=2500.0,
        gt=0,
        description=(
            "Wall-clock ceiling for a *served* request. Not 200: the organisers' "
            "bar scopes chunking and retrieval, which this system meets with "
            "room to spare (7.3 ms P50, 23.5 ms P100 measured), but a real "
            "hosted LLM call from India costs 450-900 ms of round trip that no "
            "amount of local engineering removes. Serving at 200 makes the "
            "deadline truncate every answer after roughly one word -- honest, "
            "and useless. scripts/bench_latency.py passes 200 explicitly, so "
            "the published claim is still measured against the real bar."
        ),
    )
    budget_input_guard_ms: float = Field(default=5.0, gt=0)
    budget_embed_ms: float = Field(default=10.0, gt=0)
    budget_retrieve_ms: float = Field(default=25.0, gt=0)
    budget_abstention_ms: float = Field(default=5.0, gt=0)
    budget_prompt_ms: float = Field(default=5.0, gt=0)
    budget_generate_ms: float = Field(default=2200.0, gt=0)
    budget_grounding_ms: float = Field(default=10.0, gt=0)

    # -- guardrail thresholds -------------------------------------------------

    injection_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    profanity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_query_chars: int = Field(default=600, ge=16, le=4000)
    abstain_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "P(corpus cannot answer) above which we refuse. Only used for the "
            "prior rules; a calibrated model carries its own fitted threshold."
        ),
    )
    abstain_k: int = Field(default=10, ge=1, le=100)
    guardrail_policy_file: Path | None = Field(
        default=None,
        description="JSON written by GuardrailPolicy.save(), typically a calibrated gate.",
    )
    examples_file: Path = Field(
        default=REPO_ROOT / "reports" / "examples.json",
        description=(
            "Questions verified to answer against the served index, written by "
            "scripts/verify_examples.py and offered by GET /examples. The "
            "shard's own answerable labels are a claim about the dataset, not "
            "about this pipeline, and the two disagree -- so a chip sampled "
            "from them can be one the model then declines, which is the most "
            "avoidable way to look broken on camera. When this file is absent "
            "or was generated against a different index, /examples falls back "
            "to sampling rather than failing."
        ),
    )
    grounding_claim_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    grounding_answer_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Distinct content tokens a claim needs before lexical overlap may support
    # it. At 1, coverage is a presence test: "Mumbai." scored 1.0 against
    # passages that never say Mumbai is the capital, and the fabrication was
    # served as fully grounded. See GroundingConfig.min_claim_tokens.
    grounding_min_claim_tokens: int = Field(default=2, ge=1, le=10)
    answer_on_ungrounded: bool = Field(
        default=False,
        description=(
            "Show an answer that failed grounding, flagged, instead of "
            "withholding it. Right for a research UI, wrong for a product."
        ),
    )

    # -- rate limiting --------------------------------------------------------
    # Sized for a public demo link, not for throughput. The binding external
    # constraint is Groq's free tier: 8,000 tokens/minute, and one RAG answer
    # costs roughly 1,400 (five passages of prompt plus MAX_TOKENS of output),
    # so the quota sustains about five questions a minute. The limits below sit
    # well *above* that on purpose -- their job is to stop a scraper, not to
    # ration a judge, and Groq enforces its own ceiling with a 429 that the
    # circuit breaker already handles.
    #
    # See :mod:`voicerag.api.ratelimit` for why the per-client numbers are
    # advisory and the global ones are not.

    rate_limit_enabled: bool = Field(
        default=True,
        description=(
            "Apply the limits below. On by default: the value of a limiter that "
            "has to be remembered at deploy time is zero."
        ),
    )
    rate_limit_trust_forwarded_for: bool = Field(
        default=True,
        description=(
            "Identify clients by the leftmost X-Forwarded-For entry. Required "
            "behind any platform proxy, where the socket peer is the same "
            "gateway address for every visitor. The header is client-written, "
            "so this buys granularity, not security -- the global limits are "
            "the part that cannot be spoofed."
        ),
    )
    rate_limit_max_clients: int = Field(
        default=4096,
        ge=16,
        description=(
            "Distinct identities tracked, evicted least-recently-used. Bounds "
            "the memory a client rotating X-Forwarded-For can make us allocate."
        ),
    )
    #: /ask and /ask/stream -- the only endpoints that spend LLM tokens.
    rate_limit_generate_per_min: int = Field(default=15, ge=1)
    rate_limit_generate_global_per_min: int = Field(default=45, ge=1)
    #: /speak -- Sarvam TTS, billed per character.
    rate_limit_speak_per_min: int = Field(default=30, ge=1)
    rate_limit_speak_global_per_min: int = Field(default=90, ge=1)
    #: /speculate, /examples, /stats, /stt/token. Generous because the browser
    #: fires /speculate once per transcript revision while the user is still
    #: speaking -- five to fifteen calls for a single spoken question.
    rate_limit_light_per_min: int = Field(default=240, ge=1)
    rate_limit_light_global_per_min: int = Field(default=900, ge=1)
    #: WS /stt/stream. The browser VAD opens a fresh socket per utterance, and
    #: again after each idle timeout, so the *rate* must be permissive; the
    #: concurrency caps below are what actually protect the Sarvam account.
    rate_limit_stt_per_min: int = Field(default=30, ge=1)
    rate_limit_stt_global_per_min: int = Field(default=120, ge=1)
    rate_limit_stt_concurrent: int = Field(
        default=3,
        ge=1,
        description=(
            "Simultaneous relay sessions per client. A rate limit alone cannot "
            "bound this: ten sockets opened in one second are under every "
            "per-minute limit and hold ten upstream sessions for two minutes."
        ),
    )
    rate_limit_stt_concurrent_total: int = Field(default=12, ge=1)

    # -- feature flags --------------------------------------------------------

    enable_guardrails: bool = True
    enable_grounding: bool = True
    enable_speculative_retrieval: bool = True
    enable_reranking: bool = Field(
        default=False,
        description=(
            "Cross-encoder reranking of fused hits. Off: no reranker ships in "
            "this repo, and the flag exists so the pipeline has one documented "
            "place to accept one rather than growing an ad-hoc hook later."
        ),
    )
    speculation_cache_size: int = Field(default=64, ge=1, le=4096)
    speculation_ttl_s: float = Field(default=30.0, gt=0)
    warmup_query: str = Field(
        default="what is the boiling point of water",
        description=(
            "Dummy query run at startup. Mandatory, not optional: the first "
            "request through a fresh process pays FAISS visit-table allocation, "
            "bm25s first touch and CPython call-cache warming, and a judge's "
            "first question must not be the one that pays them."
        ),
    )

    # -- validation -----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _blank_means_unset(cls, values: Any) -> Any:
        """Treat ``KEY=`` in a ``.env`` file as "not configured".

        ``.env.example`` ships every key with an empty value so it doubles as
        documentation. Without this, copying it would give ``GROQ_API_KEY`` the
        value ``""`` -- which is not ``None``, so the service would believe it
        had a credential, build a Groq client and fail on the first request with
        a 401 instead of saying plainly that no provider is configured.
        """
        if not isinstance(values, dict):
            return values
        return {k: v for k, v in values.items() if not (isinstance(v, str) and not v.strip())}

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        """Accept a comma-separated string or a JSON array.

        A shell environment variable is a string; insisting on JSON there is a
        needless source of quoting bugs. Both forms are accepted, and the JSON
        one is decoded here rather than by the settings source (see ``NoDecode``
        on the field) so there is exactly one place that knows the rule.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                import json

                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("index_dir", "guardrail_policy_file", "static_dir", "examples_file")
    @classmethod
    def _absolutise(cls, value: Path | None) -> Path | None:
        """Resolve relative paths against the repo root, not the CWD.

        ``uvicorn voicerag.api.main:app`` is routinely started from a different
        directory than the one the index was built in; silently serving an empty
        index because of that is a debugging afternoon nobody needs.
        """
        if value is None:
            return None
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    # -- derived --------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def budget_headroom_ms(self) -> float:
        """Total budget minus the sum of the stage allowances.

        Negative means the configuration cannot meet its own target even if
        every stage finishes exactly on its allowance. Surfaced by
        :meth:`public_dict` so the condition is visible in ``/healthz`` rather
        than discovered in a benchmark.
        """
        return round(self.budget_total_ms - sum(self.stage_budgets().values()), 3)

    def stage_budgets(self) -> dict[str, float]:
        """Per-stage allowance in milliseconds, keyed by trace span name.

        The keys are the span names the pipeline emits, so a stage's allowance
        and its measurement are joined on the same string and cannot drift
        apart.
        """
        return {
            "guard.input": self.budget_input_guard_ms,
            "embed": self.budget_embed_ms,
            "retrieve": self.budget_retrieve_ms,
            "guard.abstention": self.budget_abstention_ms,
            "prompt": self.budget_prompt_ms,
            "generate": self.budget_generate_ms,
            "guard.grounding": self.budget_grounding_ms,
        }

    def has_generation_provider(self) -> bool:
        """True if at least one LLM credential is configured.

        The list must match what :func:`voicerag.api.state.build_generators`
        actually constructs. It did not: OpenAI was wired there and missing here,
        so a deployment holding only an ``OPENAI_API_KEY`` -- the documented
        fallback for when Groq's free tier is exhausted -- reported itself as
        having no generation provider while serving answers perfectly well.
        """
        return (
            self.groq_api_key is not None
            or self.openai_api_key is not None
            or self.gemini_api_key is not None
        )

    def public_dict(self) -> dict[str, Any]:
        """Configuration safe to serialise into a health or stats response.

        Secrets are reduced to presence booleans. This method is the *only*
        supported way to put configuration into an HTTP response.
        """
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "index_dir": str(self.index_dir),
            "embedder_spec": self.embedder_spec,
            "chunking_strategy": self.chunking_strategy,
            "retrieval_k": self.retrieval_k,
            "fusion": self.fusion,
            "ef_search": self.ef_search,
            "max_tokens": self.max_tokens,
            "budget_total_ms": self.budget_total_ms,
            "budget_headroom_ms": self.budget_headroom_ms,
            "stage_budgets_ms": self.stage_budgets(),
            "features": {
                "guardrails": self.enable_guardrails,
                "grounding": self.enable_grounding,
                "speculative_retrieval": self.enable_speculative_retrieval,
                "reranking": self.enable_reranking,
            },
            "credentials": {
                "groq": self.groq_api_key is not None,
                "openai": self.openai_api_key is not None,
                "gemini": self.gemini_api_key is not None,
                "sarvam": self.sarvam_api_key is not None,
                "elevenlabs": self.elevenlabs_api_key is not None,
            },
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, parsed once.

    Cached because reading and validating ``.env`` on every request would be a
    measurable fraction of a 200 ms budget spent on nothing. Call
    ``get_settings.cache_clear()`` in a test that mutates the environment.
    """
    return Settings()
