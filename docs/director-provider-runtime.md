# Director Provider Runtime

Vectra's Director runtime now separates provider transport from Director control flow.

## Flow

1. The Director loop builds a provider-agnostic prompt context from the raw user prompt, scene summary, recent observations, and history.
2. `call_director()` selects provider candidates in order and resolves a provider adapter by `(family, transport)`.
3. Each adapter is responsible for:
   - building the outbound request payload
   - executing provider-specific transport
   - parsing the raw response into a normalized `ParsedProviderResponse`
4. The provider layer performs one bounded corrective retry when a response is non-actionable, then falls back to the next provider.
5. The Director loop accepts only:
   - a valid executable tool batch
   - a constrained clarification request
   - a structured failure
6. Runtime-state metadata is returned through the backend and surfaced directly in the Blender addon.

## Runtime States

- `awaiting_model_response`: the addon has dispatched a request and is waiting for the backend
- `provider_transport_failure`: the provider request failed before a usable parse result
- `tool_call_parse_failure`: the provider responded, but the tool payload was malformed
- `no_action_response`: the provider responded with narration or a non-actionable structure
- `valid_action_batch_ready`: a valid executable first-step batch is ready
- `fallback_provider_invoked`: a valid batch is ready, but it came from a fallback provider

## Boundaries

- Provider adapters own request shape, HTTP transport, and raw-response parsing.
- The Director loop owns actionability validation, reference resolution, and executable action selection.
- The Blender addon owns local in-flight UI state and displays backend runtime-state metadata without reclassifying failures.

## Extension Points

- New model backends register a new provider adapter instead of editing the Director loop.
- Future LangGraph orchestration, tool routers, retrieval memory, or model-scoring/evaluator nodes can compose around `call_director()` and the adapter registry without replacing provider transport internals.
