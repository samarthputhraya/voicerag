"""Tests for :mod:`voicerag.config`.

Configuration bugs are the quiet kind: the service starts, serves, and is wrong.
The cases here are the three that actually bite -- a copied ``.env.example``
leaving blank keys, a shell-style comma list where a JSON array was expected,
and a secret finding its way into a response body.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voicerag.config import REPO_ROOT, Settings, get_settings

ENV_EXAMPLE = REPO_ROOT / ".env.example"


def test_defaults_are_internally_feasible() -> None:
    """The stage allowances must fit inside the total, or the target is unmeetable.

    The invariant is what matters, not the constant. The serving default is no
    longer 200 ms: a real hosted LLM call from India costs 450-900 ms of round
    trip, and at a 200 ms deadline the pipeline truncates every answer after
    about one word. The 200 ms bar is still measured -- ``bench_latency.py``
    passes it explicitly -- but it is a benchmark target, not a serving config.
    """
    settings = Settings(_env_file=None)
    assert sum(settings.stage_budgets().values()) <= settings.budget_total_ms
    assert settings.budget_headroom_ms >= 0


def test_the_200ms_target_remains_feasible_for_the_benchmark() -> None:
    """The published claim must still be a configuration the system can hold.

    Retrieval-path stages have to fit inside 200 ms with the generation
    allowance the vendor decode profile implies; otherwise the headline number
    is unreachable by construction rather than by measurement.
    """
    settings = Settings(_env_file=None, budget_total_ms=200.0, budget_generate_ms=120.0)
    assert sum(settings.stage_budgets().values()) <= settings.budget_total_ms
    assert settings.budget_headroom_ms >= 0


def test_blank_env_values_mean_unset(tmp_path: Path) -> None:
    """Copying ``.env.example`` verbatim must not fake a credential into existence."""
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=\nGEMINI_API_KEY=\nSARVAM_TOKEN_URL=\nMAX_TOKENS=64\n")
    settings = Settings(_env_file=env)
    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None
    assert settings.sarvam_token_url is None
    assert settings.has_generation_provider() is False
    assert settings.max_tokens == 64


def test_env_example_parses_and_configures_nothing(tmp_path: Path) -> None:
    """The shipped example must load, and must leave the service credential-free."""
    assert ENV_EXAMPLE.exists(), "the documented configuration file is missing"
    settings = Settings(_env_file=ENV_EXAMPLE)
    assert settings.has_generation_provider() is False
    assert settings.sarvam_api_key is None
    assert settings.max_tokens == 160
    assert settings.enable_guardrails is True
    # A clone that follows the README's `cp .env.example .env` must get the
    # system as shipped, not a superseded configuration. These three drifted
    # once already and produced one-word answers on every question.
    assert settings.embedder_spec == "static:minishlab/potion-base-8M"
    assert settings.chunking_strategy == "recursive"
    # Generous on purpose. Measured, real Groq calls from India land between
    # 0.5 s and 4.4 s wall clock; a tighter serving deadline truncates the
    # slower ones mid-sentence and hands back the fragment as a cited answer.
    assert settings.budget_total_ms >= 5000.0
    del tmp_path


def test_env_example_documents_every_field() -> None:
    """Every setting must appear in ``.env.example``; undocumented knobs get lost."""
    text = ENV_EXAMPLE.read_text()
    missing = [
        name
        for name in Settings.model_fields
        if f"\n{name.upper()}=" not in text
    ]
    assert missing == [], f"undocumented settings: {missing}"


def test_cors_origins_accepts_both_forms() -> None:
    assert Settings(_env_file=None, cors_origins="https://a, https://b").cors_origins == [
        "https://a",
        "https://b",
    ]
    assert Settings(_env_file=None, cors_origins='["https://c"]').cors_origins == ["https://c"]
    assert Settings(_env_file=None, cors_origins=["https://d"]).cors_origins == ["https://d"]


def test_relative_paths_resolve_against_the_repo_not_the_cwd() -> None:
    """``uvicorn`` is routinely launched from elsewhere; the index must still be found."""
    settings = Settings(_env_file=None, index_dir="data/index")
    assert settings.index_dir == (REPO_ROOT / "data" / "index").resolve()
    assert settings.index_dir.is_absolute()


def test_public_dict_never_contains_a_secret() -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="gsk-secret",
        sarvam_api_key="sk-secret",
        stt_signing_key="hmac-secret",
    )
    rendered = repr(settings.public_dict())
    assert "gsk-secret" not in rendered
    assert "sk-secret" not in rendered
    assert "hmac-secret" not in rendered
    assert settings.public_dict()["credentials"] == {
        "groq": True,
        "openai": False,
        "gemini": False,
        "sarvam": True,
        "elevenlabs": False,
    }


def test_out_of_range_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_tokens=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, abstain_threshold=1.5)
    with pytest.raises(ValueError):
        Settings(_env_file=None, fusion="not-a-fusion-method")


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
