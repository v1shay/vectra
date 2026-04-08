from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_DIRECTOR_TIMEOUT_SECONDS = 45.0
DEFAULT_DIRECTOR_MAX_RETRIES = 2
DEFAULT_DIRECTOR_STEP_DEADLINE_SECONDS = 60.0
DEFAULT_DIRECTOR_PROVIDER_ATTEMPT_BUDGET = 4
DEFAULT_CONTROLLER_TIMEOUT_SECONDS = 12.0
DEFAULT_CONTROLLER_MAX_RETRIES = 1
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_PRIMARY = "qwen2.5-coder:32b"
DEFAULT_OLLAMA_SECONDARY = "deepseek-coder-v2:16b"
DEFAULT_DIRECTOR_MODEL = "gpt-5.1"
DEFAULT_CONTROLLER_MODEL = "grok-4.2-reasoning"
DEFAULT_DIRECTOR_TRANSPORT = "responses"
DEFAULT_CONTROLLER_TRANSPORT = "responses"


@dataclass(frozen=True)
class EndpointConfig:
    provider: str
    family: str
    base_url: str
    api_key: str | None
    model: str
    transport: str = "responses"


@dataclass(frozen=True)
class RuntimeConfig:
    controller: EndpointConfig | None
    director: EndpointConfig | None
    director_timeout_seconds: float
    director_max_retries: int
    director_step_deadline_seconds: float
    director_provider_attempt_budget: int
    controller_timeout_seconds: float
    controller_max_retries: int
    ollama_host: str
    ollama_primary_model: str
    ollama_secondary_model: str


def _normalize_http_url(raw_value: str) -> str:
    normalized = raw_value.strip()
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(float(raw_value), 1.0)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), 0)
    except ValueError:
        return default


def _read_endpoint(
    *,
    provider: str,
    family: str,
    base_url_vars: tuple[str, ...],
    api_key_vars: tuple[str, ...],
    model_vars: tuple[str, ...],
    default_model: str,
    transport: str = "responses",
    transport_vars: tuple[str, ...] = (),
) -> EndpointConfig | None:
    base_url = ""
    api_key = ""
    model = ""
    for var_name in base_url_vars:
        candidate = _normalize_http_url(os.getenv(var_name, ""))
        if candidate:
            base_url = candidate
            break
    for var_name in api_key_vars:
        candidate = os.getenv(var_name, "").strip()
        if candidate:
            api_key = candidate
            break
    for var_name in model_vars:
        candidate = os.getenv(var_name, "").strip()
        if candidate:
            model = candidate
            break
    for var_name in transport_vars:
        candidate = os.getenv(var_name, "").strip().lower()
        if candidate:
            transport = candidate
            break
    if not model:
        model = default_model
    if not (base_url and api_key):
        return None
    return EndpointConfig(
        provider=provider,
        family=family,
        base_url=base_url,
        api_key=api_key,
        model=model,
        transport=transport,
    )


def load_runtime_config() -> RuntimeConfig:
    controller = _read_endpoint(
        provider="xai-controller",
        family="xai",
        base_url_vars=("VECTRA_CONTROLLER_BASE_URL",),
        api_key_vars=("VECTRA_CONTROLLER_API_KEY",),
        model_vars=("VECTRA_CONTROLLER_MODEL",),
        default_model=DEFAULT_CONTROLLER_MODEL,
        transport=DEFAULT_CONTROLLER_TRANSPORT,
        transport_vars=("VECTRA_CONTROLLER_TRANSPORT",),
    )
    director = _read_endpoint(
        provider="openai-director",
        family="openai",
        base_url_vars=("VECTRA_DIRECTOR_BASE_URL", "VECTRA_LLM_BASE_URL"),
        api_key_vars=("VECTRA_DIRECTOR_API_KEY", "VECTRA_LLM_API_KEY"),
        model_vars=("VECTRA_DIRECTOR_MODEL", "VECTRA_LLM_MODEL"),
        default_model=DEFAULT_DIRECTOR_MODEL,
        transport=DEFAULT_DIRECTOR_TRANSPORT,
        transport_vars=("VECTRA_DIRECTOR_TRANSPORT",),
    )
    return RuntimeConfig(
        controller=controller,
        director=director,
        director_timeout_seconds=_read_float_env(
            "VECTRA_DIRECTOR_TIMEOUT_SECONDS",
            _read_float_env("VECTRA_LLM_TIMEOUT_SECONDS", DEFAULT_DIRECTOR_TIMEOUT_SECONDS),
        ),
        director_max_retries=_read_int_env(
            "VECTRA_DIRECTOR_MAX_RETRIES",
            _read_int_env("VECTRA_LLM_MAX_RETRIES", DEFAULT_DIRECTOR_MAX_RETRIES),
        ),
        director_step_deadline_seconds=_read_float_env(
            "VECTRA_DIRECTOR_STEP_DEADLINE_SECONDS",
            DEFAULT_DIRECTOR_STEP_DEADLINE_SECONDS,
        ),
        director_provider_attempt_budget=max(
            _read_int_env(
                "VECTRA_DIRECTOR_PROVIDER_ATTEMPT_BUDGET",
                DEFAULT_DIRECTOR_PROVIDER_ATTEMPT_BUDGET,
            ),
            1,
        ),
        controller_timeout_seconds=_read_float_env(
            "VECTRA_CONTROLLER_TIMEOUT_SECONDS",
            DEFAULT_CONTROLLER_TIMEOUT_SECONDS,
        ),
        controller_max_retries=_read_int_env(
            "VECTRA_CONTROLLER_MAX_RETRIES",
            DEFAULT_CONTROLLER_MAX_RETRIES,
        ),
        ollama_host=_normalize_http_url(os.getenv("OLLAMA_HOST", "") or DEFAULT_OLLAMA_HOST),
        ollama_primary_model=os.getenv(
            "VECTRA_OLLAMA_MODEL_PRIMARY",
            "",
        ).strip()
        or os.getenv("VECTRA_OLLAMA_MODEL", "").strip()
        or DEFAULT_OLLAMA_PRIMARY,
        ollama_secondary_model=os.getenv(
            "VECTRA_OLLAMA_MODEL_SECONDARY",
            "",
        ).strip()
        or DEFAULT_OLLAMA_SECONDARY,
    )
