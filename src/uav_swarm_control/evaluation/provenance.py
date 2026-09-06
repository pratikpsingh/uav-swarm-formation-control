"""Shared source and runtime provenance for reproducible experiment runners."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch


def _installed_version(distribution: str) -> str:
    """Report an optional distribution without making it a runtime dependency."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def stable_digest(value: object) -> str:
    """Hash a JSON-compatible value with deterministic key ordering."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_files(project_root: Path) -> list[Path]:
    """Return the project-owned files that define an experiment execution."""
    return [
        *sorted((project_root / "src").rglob("*.py")),
        project_root / "pyproject.toml",
        project_root / "uv.lock",
    ]


def collect_provenance(project_root: Path) -> dict[str, object]:
    """Capture code hashes, revision, runtime versions, platform, and thread count."""
    hashes = {
        str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files(project_root)
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "git_revision": revision,
        "git_dirty": bool(dirty),
        "source_files_sha256": hashes,
        "source_sha256": stable_digest(hashes),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchao": _installed_version("torchao"),
        "numpy": _installed_version("numpy"),
        "pybullet": _installed_version("pybullet"),
        "scipy": _installed_version("scipy"),
        "torch_threads": torch.get_num_threads(),
    }


def write_source_snapshot(project_root: Path, output: Path) -> None:
    """Archive the exact source and lock files used by an experiment."""
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in source_files(project_root):
            archive.write(path, str(path.relative_to(project_root)))
