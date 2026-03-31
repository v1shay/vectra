from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SOURCE_DIR_NAME = "vectra"
ARCHIVE_ROOT_NAME = "vectra"
OUTPUT_NAME = "vectra_addon.zip"


def iter_addon_files(source_dir: Path):
    for path in sorted(source_dir.rglob("*")):
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        yield path


def build_archive_path(source_dir: Path, file_path: Path) -> Path:
    relative_path = file_path.relative_to(source_dir)
    return Path(ARCHIVE_ROOT_NAME) / relative_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / SOURCE_DIR_NAME
    output_path = repo_root / OUTPUT_NAME

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source add-on directory: {source_dir}")

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in iter_addon_files(source_dir):
            archive.write(file_path, build_archive_path(source_dir, file_path))

    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
