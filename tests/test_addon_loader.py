from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vectra import addon_loader


def _package_module(package_name: str, packaged_dir: Path) -> ModuleType:
    package_module = ModuleType(package_name)
    package_module.__file__ = str(packaged_dir / "__init__.py")
    package_module.__package__ = package_name
    package_module.__path__ = [str(packaged_dir)]
    package_module.__spec__ = SimpleNamespace(
        submodule_search_locations=[str(packaged_dir)]
    )
    return package_module


def test_resolve_dev_source_path_accepts_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "vectra"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    resolved, error = addon_loader.resolve_dev_source_path(str(repo_root))

    assert resolved == repo_root.resolve()
    assert error is None


def test_resolve_dev_source_path_rejects_invalid_repo_root(tmp_path: Path) -> None:
    resolved, error = addon_loader.resolve_dev_source_path(str(tmp_path / "missing"))

    assert resolved is None
    assert error == f"Invalid Vectra dev source path: {(tmp_path / 'missing').resolve()}"


def test_resolve_dev_source_path_ignores_non_string_values() -> None:
    resolved, error = addon_loader.resolve_dev_source_path(object())

    assert resolved is None
    assert error is None


def test_runtime_namespace_uses_full_dotted_package_name(tmp_path: Path) -> None:
    packaged_dir = tmp_path / "packaged" / "vectra"
    packaged_dir.mkdir(parents=True)
    package_module = _package_module("bl_ext.user_default.vectra", packaged_dir)

    namespace = addon_loader._runtime_namespace(package_module)

    assert namespace.package_name == "bl_ext.user_default.vectra"
    assert namespace.runtime_module_name == "bl_ext.user_default.vectra.addon_runtime"
    assert namespace.stable_modules == frozenset(
        {
            "bl_ext.user_default.vectra",
            "bl_ext.user_default.vectra.addon_bootstrap",
            "bl_ext.user_default.vectra.addon_loader",
        }
    )


def test_clear_managed_modules_preserves_stable_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    packaged_dir = tmp_path / "packaged" / "vectra"
    packaged_dir.mkdir(parents=True)
    package_module = _package_module("bl_ext.user_default.vectra", packaged_dir)
    namespace = addon_loader._runtime_namespace(package_module)
    fake_modules = {
        "bl_ext.user_default.vectra": ModuleType("bl_ext.user_default.vectra"),
        "bl_ext.user_default.vectra.addon_loader": ModuleType("bl_ext.user_default.vectra.addon_loader"),
        "bl_ext.user_default.vectra.addon_bootstrap": ModuleType("bl_ext.user_default.vectra.addon_bootstrap"),
        "bl_ext.user_default.vectra.addon_runtime": ModuleType("bl_ext.user_default.vectra.addon_runtime"),
        "bl_ext.user_default.vectra.tools.mesh_tools": ModuleType("bl_ext.user_default.vectra.tools.mesh_tools"),
        "bl_ext.user_default.vectra.ui.panel": ModuleType("bl_ext.user_default.vectra.ui.panel"),
        "other.module": ModuleType("other.module"),
    }
    monkeypatch.setattr(addon_loader, "sys", SimpleNamespace(modules=fake_modules))

    addon_loader._clear_managed_modules(namespace)

    assert sorted(fake_modules) == [
        "bl_ext.user_default.vectra",
        "bl_ext.user_default.vectra.addon_bootstrap",
        "bl_ext.user_default.vectra.addon_loader",
        "other.module",
    ]


def test_activate_runtime_prefers_dev_source_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    packaged_dir = tmp_path / "packaged" / "vectra"
    packaged_dir.mkdir(parents=True)
    source_dir = tmp_path / "repo" / "vectra"
    source_dir.mkdir(parents=True)
    (source_dir / "__init__.py").write_text("", encoding="utf-8")

    package_module = _package_module("bl_ext.user_default.vectra", packaged_dir)
    fake_runtime = SimpleNamespace(register=lambda: None)
    requested_modules: list[str] = []

    monkeypatch.setattr(addon_loader, "_shutdown_current_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(addon_loader, "_clear_managed_modules", lambda namespace: None)
    monkeypatch.setattr(addon_loader.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(
        addon_loader.importlib,
        "import_module",
        lambda name: requested_modules.append(name) or fake_runtime,
    )
    addon_loader.reset_loader_state()

    runtime_module = addon_loader.activate_runtime(
        package_module,
        dev_source_path=str(tmp_path / "repo"),
    )

    assert runtime_module is fake_runtime
    assert package_module.__path__ == [str(source_dir), str(packaged_dir)]
    assert package_module.__spec__.submodule_search_locations == [
        str(source_dir),
        str(packaged_dir),
    ]
    assert requested_modules == ["bl_ext.user_default.vectra.addon_runtime"]
    assert addon_loader.get_runtime_status() == addon_loader.RuntimeStatus(
        mode="dev-source",
        source_path=str((tmp_path / "repo").resolve()),
        error=None,
    )


def test_activate_runtime_falls_back_to_packaged_on_invalid_dev_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged_dir = tmp_path / "packaged" / "vectra"
    packaged_dir.mkdir(parents=True)
    package_module = _package_module("bl_ext.user_default.vectra", packaged_dir)
    fake_runtime = SimpleNamespace(register=lambda: None)

    monkeypatch.setattr(addon_loader, "_shutdown_current_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(addon_loader, "_clear_managed_modules", lambda namespace: None)
    monkeypatch.setattr(addon_loader.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(addon_loader.importlib, "import_module", lambda name: fake_runtime)
    addon_loader.reset_loader_state()

    runtime_module = addon_loader.activate_runtime(
        package_module,
        dev_source_path=str(tmp_path / "missing"),
    )

    assert runtime_module is fake_runtime
    assert package_module.__path__ == [str(packaged_dir)]
    status = addon_loader.get_runtime_status()
    assert status.mode == "packaged"
    assert status.source_path is None
    assert status.error == f"Invalid Vectra dev source path: {(tmp_path / 'missing').resolve()}"
