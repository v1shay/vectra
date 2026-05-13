from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from .adapters import ProviderRequest, get_provider_adapter, register_default_provider_adapters
from .config import EndpointConfig, RuntimeConfig, load_runtime_config


DEFAULT_AI_PROBE_TIMEOUT_SECONDS = 5.0


def _endpoint_summary(endpoint: EndpointConfig | None, *, role: str) -> dict[str, Any]:
    if endpoint is None:
        return {"role": role, "configured": False}
    return {
        "role": role,
        "configured": True,
        "provider": endpoint.provider,
        "family": endpoint.family,
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "transport": endpoint.transport,
        "api_key_configured": bool(endpoint.api_key),
    }


def _runtime_settings_summary(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "director_timeout_seconds": runtime.director_timeout_seconds,
        "director_max_retries": runtime.director_max_retries,
        "director_step_deadline_seconds": runtime.director_step_deadline_seconds,
        "director_provider_attempt_budget": runtime.director_provider_attempt_budget,
        "controller_timeout_seconds": runtime.controller_timeout_seconds,
        "controller_max_retries": runtime.controller_max_retries,
        "audit_mode": runtime.audit_mode,
        "disable_provider_fallback": runtime.disable_provider_fallback,
        "ollama_host": runtime.ollama_host,
        "ollama_primary_model": runtime.ollama_primary_model,
        "ollama_secondary_model": runtime.ollama_secondary_model,
    }


def configured_ai_summary(runtime: RuntimeConfig | None = None) -> dict[str, Any]:
    runtime = runtime or load_runtime_config()
    providers = [
        _endpoint_summary(runtime.director, role="director"),
        _endpoint_summary(runtime.controller, role="controller"),
        {
            "role": "local_fallback",
            "configured": bool(runtime.ollama_host),
            "provider": "ollama-primary",
            "family": "ollama",
            "base_url": runtime.ollama_host,
            "model": runtime.ollama_primary_model,
            "transport": "ollama_json_envelope",
            "api_key_configured": False,
        },
    ]
    return {
        "configured": any(provider.get("configured") for provider in providers),
        "providers": providers,
        "settings": _runtime_settings_summary(runtime),
    }


def _primary_probe_endpoint(runtime: RuntimeConfig) -> EndpointConfig | None:
    if runtime.director is not None:
        return runtime.director
    if runtime.controller is not None:
        return runtime.controller
    if runtime.ollama_host:
        return EndpointConfig(
            provider="ollama-primary",
            family="ollama",
            base_url=runtime.ollama_host,
            api_key=None,
            model=runtime.ollama_primary_model,
            transport="ollama_json_envelope",
        )
    return None


def _probe_endpoint(endpoint: EndpointConfig, *, timeout_seconds: float) -> dict[str, Any]:
    register_default_provider_adapters()
    started = time.perf_counter()
    try:
        adapter = get_provider_adapter(endpoint)
        result = adapter.invoke(
            endpoint,
            ProviderRequest(
                instructions=(
                    "You are a Vectra runtime health probe. Reply with exactly "
                    "'vectra-ai-ok'. Do not call tools."
                ),
                user_input="Health probe: reply with vectra-ai-ok.",
                tools=[],
            ),
            timeout=max(min(float(timeout_seconds), DEFAULT_AI_PROBE_TIMEOUT_SECONDS), 1.0),
            max_retries=0,
        )
    except Exception as exc:
        return {
            "status": "error",
            "provider": endpoint.provider,
            "model": endpoint.model,
            "transport": endpoint.transport,
            "runtime_state": getattr(exc, "runtime_state", "provider_transport_failure"),
            "failure_reason": str(exc) or exc.__class__.__name__,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    attempt = result.attempt
    assistant_text = result.parsed.assistant_text if result.parsed is not None else ""
    transport_ok = bool(result.parsed is not None)
    return {
        "status": "ok" if transport_ok else "error",
        "provider": endpoint.provider,
        "model": endpoint.model,
        "transport": endpoint.transport,
        "runtime_state": "ai_probe_succeeded" if transport_ok else attempt.runtime_state,
        "response_type": attempt.response_type,
        "assistant_text_preview": assistant_text[:80],
        "failure_reason": "" if transport_ok else attempt.failure_reason,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "attempt": asdict(attempt),
    }


def run_ai_diagnostics(
    *,
    probe: bool = True,
    timeout_seconds: float = DEFAULT_AI_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    runtime = load_runtime_config()
    summary = configured_ai_summary(runtime)
    primary = _primary_probe_endpoint(runtime)
    if primary is None:
        summary["status"] = "unconfigured"
        summary["probe"] = {
            "status": "skipped",
            "runtime_state": "unconfigured",
            "failure_reason": "No director or controller provider is configured.",
        }
        return summary

    if not probe:
        summary["status"] = "configured" if summary["configured"] else "unconfigured"
        summary["probe"] = {"status": "skipped"}
        return summary

    probe_result = _probe_endpoint(primary, timeout_seconds=timeout_seconds)
    summary["probe"] = probe_result
    summary["status"] = "ok" if probe_result["status"] == "ok" else "error"
    return summary
