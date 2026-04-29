from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .director.config import RuntimeConfig, load_runtime_config
from .director.models import ControllerDecision, ProviderResult
from .director.providers import ProviderError, ProviderTimeoutError, call_controller, call_director


class LLMClientError(Exception):
    """Base exception for provider failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when the runtime has no usable provider configuration."""


class LLMRequestError(LLMClientError):
    """Raised when a provider request cannot be completed."""


class LLMTimeoutError(LLMRequestError):
    """Raised when the provider chain times out."""


@dataclass(frozen=True)
class LLMRuntimeSettings:
    director_timeout_seconds: float
    director_max_retries: int
    director_step_deadline_seconds: float
    director_provider_attempt_budget: int
    controller_timeout_seconds: float
    controller_max_retries: int
    ollama_host: str
    ollama_primary_model: str
    ollama_secondary_model: str
    audit_mode: bool = False
    disable_provider_fallback: bool = False


def read_runtime_settings() -> LLMRuntimeSettings:
    runtime = load_runtime_config()
    return LLMRuntimeSettings(
        director_timeout_seconds=runtime.director_timeout_seconds,
        director_max_retries=runtime.director_max_retries,
        director_step_deadline_seconds=runtime.director_step_deadline_seconds,
        director_provider_attempt_budget=runtime.director_provider_attempt_budget,
        controller_timeout_seconds=runtime.controller_timeout_seconds,
        controller_max_retries=runtime.controller_max_retries,
        ollama_host=runtime.ollama_host,
        ollama_primary_model=runtime.ollama_primary_model,
        ollama_secondary_model=runtime.ollama_secondary_model,
        audit_mode=runtime.audit_mode,
        disable_provider_fallback=runtime.disable_provider_fallback,
    )


def load_provider_config() -> RuntimeConfig:
    return load_runtime_config()


def route_prompt(prompt: str, scene_state: dict[str, Any]) -> ControllerDecision:
    try:
        return call_controller(prompt, scene_state)
    except Exception as exc:  # pragma: no cover - defensive compatibility guard
        raise LLMRequestError(str(exc)) from exc


def request_director_turn(*, prompt_text: str, tools: list[dict[str, Any]]) -> ProviderResult:
    runtime = load_runtime_config()
    if runtime.director is None and runtime.controller is None and not runtime.ollama_host:
        raise LLMConfigurationError("No director, controller, or local fallback provider is configured.")
    try:
        return call_director(prompt_text=prompt_text, tools=tools)
    except ProviderTimeoutError as exc:
        raise LLMTimeoutError(str(exc)) from exc
    except ProviderError as exc:
        raise LLMRequestError(str(exc)) from exc


def extract_scene_intent(prompt: str, scene_state: dict[str, Any]) -> None:
    del prompt, scene_state
    raise LLMClientError("The SceneIntent pipeline has been retired in favor of the Director loop.")
