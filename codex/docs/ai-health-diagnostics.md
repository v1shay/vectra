# AI Health Diagnostics

Vectra now separates backend liveness from AI/provider liveness.

## Fast Checks

```bash
.venv/bin/python - <<'PY'
from fastapi.testclient import TestClient
from agent_runtime.main import app

client = TestClient(app)
print(client.get("/health").json())
print(client.get("/ai/health?probe=false").json())
PY
```

- `/health` proves the FastAPI runtime is reachable.
- `/ai/health?probe=false` reports redacted provider configuration without making a model request.
- `/ai/health?probe=true&timeout_seconds=5` makes one short probe request to the active provider path and reports provider, model, transport, runtime state, elapsed time, and failure reason.

## Blender UI

The Vectra panel has a `Test AI` button next to `Start Backend`.

`Test AI` starts the managed backend if needed, then calls `/ai/health` with the short probe. It does not start the scene-generation loop, so it is the first check to run when the UI is stuck at `awaiting_model_response` or a model call has been taking minutes.

## Configuration Notes

Runtime secrets still belong in ignored local config such as `.vectra/runtime.env`.

Supported director aliases now include:

- `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, optional `OPENAI_MODEL`
- `NVIDIA_API_KEY`, optional `NVIDIA_BASE_URL`, optional `NVIDIA_MODEL`
- existing `VECTRA_DIRECTOR_*` and `VECTRA_LLM_*` variables

OpenAI-compatible providers can use `VECTRA_DIRECTOR_TRANSPORT=chat_completions` or `responses` as appropriate.

## Verification From This Pass

```bash
.venv/bin/pytest tests/test_backend.py tests/test_director_runtime_states.py tests/test_addon_runtime.py
.venv/bin/python -m compileall agent_runtime vectra
```

Result: focused tests passed, and compileall completed.
