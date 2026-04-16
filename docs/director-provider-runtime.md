# Director Provider Runtime

Vectra's Director runtime now separates provider transport from Director control flow.

## Flow

1. The Director loop builds a provider-agnostic prompt context from the raw user prompt, scene summary, recent observations, and history.
2. `call_director()` selects provider candidates in order and resolves a provider adapter by `(family, transport)`.
3. Each adapter is responsible for:
   - building the outbound request payload
   - executing provider-specific transport
   - parsing the raw response into a normalized `ParsedProviderResponse`
4. The HTTP adapter retries retryable transport failures up to the configured provider retry limit, then returns a failed provider attempt with transport metadata.
5. The provider layer performs one bounded corrective retry when a response is non-actionable, then falls back to the next provider.
6. The provider chain is bounded by a step-level deadline and a provider-attempt budget, so fallback cannot silently burn the whole run.
7. The Director loop accepts only:
   - a valid executable tool batch
   - a constrained clarification request
   - a structured failure
8. The Director loop validates tool names and arguments before execution, performs one corrective retry when a tool batch is not executable, and only then surfaces a failure.
9. Runtime-state metadata is returned through the backend and surfaced directly in the Blender addon.

## HTTP Retry Policy

Retryable failures:

- HTTP `429`, `500`, `502`, `503`, and `504`
- network/request interruptions
- provider timeouts, within the configured retry and step-deadline budget
- JSON decode failures or non-object JSON payloads

Fail-fast failures:

- HTTP `400`, `401`, `403`, and `404`

Every provider attempt records transport metadata when available: provider, model, transport, status code, retry count, elapsed time, failure reason, request payload preview, response/error payload preview, and per-transport-attempt entries. A retryable primary-provider failure should not kill the loop by itself; once its retry budget is exhausted, `call_director()` may continue to the configured fallback providers.

## Runtime States

- `awaiting_model_response`: the addon has dispatched a request and is waiting for the backend
- `provider_transport_failure`: the provider request failed before a usable parse result
- `provider_deadline_exceeded`: the provider request chain ran out of time before yielding a usable batch
- `tool_call_parse_failure`: the provider responded, but the tool payload was malformed
- `tool_validation_failure`: the provider returned tool names or arguments that do not match the live tool surface
- `no_action_response`: the provider responded with narration or a non-actionable structure
- `valid_action_batch_ready`: a valid executable first-step batch is ready
- `fallback_provider_invoked`: a valid batch is ready, but it came from a fallback provider

## Boundaries

- Provider adapters own request shape, HTTP transport, and raw-response parsing.
- The Director loop owns actionability validation, tool-surface validation, reference resolution, and executable action selection.
- The Blender addon owns local in-flight UI state and displays backend runtime-state metadata without reclassifying failures.
- The Blender addon can health-check/start only the local managed backend through `vectra.start_backend`; it does not start remote, SSH, or cloud processes.

## Blender Backend Control

The Vectra panel exposes a local `Start Backend` control. The operator:

- calls the existing local `ensure_local_backend()` path
- uses the configured Development Source Path when present
- updates `vectra_backend_status` as `starting`, `online`, or `failed`
- displays the expected or active `.vectra/backend.log` path when it can be resolved
- fails visibly with the same actionable missing-repo or missing-venv guidance as the runtime service

## Extension Points

- New model backends register a new provider adapter instead of editing the Director loop.
- Future LangGraph orchestration, tool routers, retrieval memory, or model-scoring/evaluator nodes can compose around `call_director()` and the adapter registry without replacing provider transport internals.

## IMPORTANT TO UNDERSTAND AS U CODE

- Each new file you edit or create should be indivually commited
- If you are given a task and it requires editing of 20 files, you should create a commit and push to GitHub after each file you edit, meaning I should recieve both 20 commits on my GitHub, and they should be 100 percent authored by me into the branch we are working on so it as if they are fully my work, meaning I will not have to merge anything the way normal codex pushes work
