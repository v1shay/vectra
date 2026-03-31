from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_PACKAGED_PACKAGE_DIR = Path(__file__).resolve().parent
_ACTIVE_MODE = "packaged"
_ACTIVE_SOURCE_PATH: str | None = None
_LAST_ERROR: str | None = None


class RuntimeLoadError(Exception):
    """Raised when the add-on runtime cannot be loaded."""


class RuntimeReloadBlockedError(RuntimeLoadError):
    """Raised when reload is unsafe because runtime work is still active."""


@dataclass(frozen=True)
class RuntimeStatus:
    mode: str
    source_path: str | None
    error: str | None


@dataclass(frozen=True)
class RuntimeNamespace:
    package_name: str
    packaged_package_dir: Path
    package_dir_name: str
    runtime_module_name: str
    stable_modules: frozenset[str]
    reload_managed_roots: tuple[str, ...]


def get_runtime_status() -> RuntimeStatus:
    return RuntimeStatus(
        mode=_ACTIVE_MODE,
        source_path=_ACTIVE_SOURCE_PATH,
        error=_LAST_ERROR,
    )


def reset_loader_state() -> None:
    global _ACTIVE_MODE, _ACTIVE_SOURCE_PATH, _LAST_ERROR

    _ACTIVE_MODE = "packaged"
    _ACTIVE_SOURCE_PATH = None
    _LAST_ERROR = None


def _package_name_for(package_module: ModuleType) -> str:
    raw_package_name = getattr(package_module, "__package__", None)
    if not isinstance(raw_package_name, str) or not raw_package_name.strip():
        raw_package_name = getattr(package_module, "__name__", "")

    package_name = raw_package_name.strip()
    if not package_name:
        raise RuntimeLoadError("Unable to determine the active Vectra add-on package name")
    return package_name


def _packaged_package_dir_for(package_module: ModuleType) -> Path:
    raw_file = getattr(package_module, "__file__", None)
    if isinstance(raw_file, str) and raw_file:
        return Path(raw_file).resolve().parent
    return _PACKAGED_PACKAGE_DIR


def _runtime_namespace(package_module: ModuleType) -> RuntimeNamespace:
    package_name = _package_name_for(package_module)
    packaged_package_dir = _packaged_package_dir_for(package_module)
    package_dir_name = packaged_package_dir.name
    return RuntimeNamespace(
        package_name=package_name,
        packaged_package_dir=packaged_package_dir,
        package_dir_name=package_dir_name,
        runtime_module_name=f"{package_name}.addon_runtime",
        stable_modules=frozenset(
            {
                package_name,
                f"{package_name}.addon_bootstrap",
                f"{package_name}.addon_loader",
            }
        ),
        reload_managed_roots=(
            f"{package_name}.addon_runtime",
            f"{package_name}.bridge",
            f"{package_name}.execution",
            f"{package_name}.operators",
            f"{package_name}.tools",
            f"{package_name}.ui",
            f"{package_name}.utils",
        ),
    )


def resolve_dev_source_path(
    raw_path: Any,
    *,
    package_dir_name: str | None = None,
) -> tuple[Path | None, str | None]:
    if raw_path is None:
        return None, None
    if not isinstance(raw_path, str):
        return None, None

    normalized = raw_path.strip()
    if not normalized:
        return None, None

    repo_root = Path(normalized).expanduser().resolve()
    package_dir = repo_root / (package_dir_name or _PACKAGED_PACKAGE_DIR.name)
    init_file = package_dir / "__init__.py"
    if not package_dir.is_dir() or not init_file.is_file():
        return None, f"Invalid Vectra dev source path: {repo_root}"
    return repo_root, None


def _managed_module_names(namespace: RuntimeNamespace) -> list[str]:
    managed = [
        name
        for name in sys.modules
        if name not in namespace.stable_modules
        and any(
            name == root or name.startswith(root + ".")
            for root in namespace.reload_managed_roots
        )
    ]
    return sorted(managed, key=lambda name: (name.count("."), name), reverse=True)


def _clear_managed_modules(namespace: RuntimeNamespace) -> None:
    for module_name in _managed_module_names(namespace):
        sys.modules.pop(module_name, None)


def _set_package_search_paths(package_module: ModuleType, search_paths: list[str]) -> None:
    package_module.__path__ = search_paths

    package_spec = getattr(package_module, "__spec__", None)
    if package_spec is None:
        return

    search_locations = getattr(package_spec, "submodule_search_locations", None)
    if search_locations is None:
        package_spec.submodule_search_locations = search_paths
        return

    try:
        search_locations[:] = search_paths
    except TypeError:
        package_spec.submodule_search_locations = search_paths


def _search_paths_for(namespace: RuntimeNamespace, repo_root: Path | None) -> list[str]:
    packaged_path = str(namespace.packaged_package_dir)
    if repo_root is None:
        return [packaged_path]

    source_path = str(repo_root / namespace.package_dir_name)
    if source_path == packaged_path:
        return [packaged_path]
    return [source_path, packaged_path]


def _shutdown_current_runtime(namespace: RuntimeNamespace, *, enforce_idle: bool) -> None:
    runtime_module = sys.modules.get(namespace.runtime_module_name)
    if runtime_module is None:
        return

    if enforce_idle:
        reason_getter = getattr(runtime_module, "get_reload_block_reason", None)
        if callable(reason_getter):
            reason = reason_getter()
            if reason:
                raise RuntimeReloadBlockedError(reason)

    unregister = getattr(runtime_module, "unregister", None)
    if not callable(unregister):
        return

    try:
        unregister()
    except Exception as exc:
        raise RuntimeLoadError(f"Failed to shut down existing Vectra runtime: {exc}") from exc


def _load_runtime_module(package_module: ModuleType, repo_root: Path | None) -> ModuleType:
    namespace = _runtime_namespace(package_module)
    importlib.invalidate_caches()
    _set_package_search_paths(package_module, _search_paths_for(namespace, repo_root))

    runtime_module = importlib.import_module(namespace.runtime_module_name)
    register = getattr(runtime_module, "register", None)
    if not callable(register):
        raise RuntimeLoadError(
            f"Runtime module '{namespace.runtime_module_name}' is missing register()"
        )

    register()
    return runtime_module


def activate_runtime(package_module: ModuleType, *, dev_source_path: str | None) -> ModuleType:
    global _ACTIVE_MODE, _ACTIVE_SOURCE_PATH, _LAST_ERROR

    namespace = _runtime_namespace(package_module)
    repo_root, validation_error = resolve_dev_source_path(
        dev_source_path,
        package_dir_name=namespace.package_dir_name,
    )
    _shutdown_current_runtime(namespace, enforce_idle=True)
    _clear_managed_modules(namespace)

    candidates: list[tuple[str, Path | None]] = []
    if repo_root is not None:
        candidates.append(("dev-source", repo_root))
    candidates.append(("packaged", None))

    last_error = validation_error
    last_exception: Exception | None = None

    for mode, candidate_root in candidates:
        try:
            _clear_managed_modules(namespace)
            runtime_module = _load_runtime_module(package_module, candidate_root)
        except Exception as exc:
            last_exception = exc
            if mode == "dev-source":
                last_error = f"Dev reload failed; using packaged runtime instead: {exc}"
                continue
            _LAST_ERROR = last_error or str(exc)
            raise RuntimeLoadError(_LAST_ERROR) from exc

        _ACTIVE_MODE = mode
        _ACTIVE_SOURCE_PATH = str(candidate_root) if candidate_root is not None else None
        _LAST_ERROR = last_error if mode == "packaged" else None
        return runtime_module

    message = last_error or "Unable to load the Vectra runtime"
    if last_exception is not None:
        raise RuntimeLoadError(message) from last_exception
    raise RuntimeLoadError(message)


def deactivate_runtime(package_module: ModuleType, *, enforce_idle: bool = False) -> None:
    namespace = _runtime_namespace(package_module)
    _shutdown_current_runtime(namespace, enforce_idle=enforce_idle)
    _clear_managed_modules(namespace)
    importlib.invalidate_caches()
    _set_package_search_paths(package_module, _search_paths_for(namespace, None))
    reset_loader_state()
