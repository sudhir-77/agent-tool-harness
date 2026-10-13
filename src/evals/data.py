"""Dataset cache, checksums, and the scripted MiniLM config download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from harness.registry import ToolSpec

MINILM_CONFIG_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
)
MINILM_CONFIG_SHA256 = "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_task_split(tasks: list[dict[str, Any]], n_tools: int) -> dict[str, int]:
    if n_tools < 300:
        raise RuntimeError(f"expected at least 300 tools, found {n_tools}")
    if len(tasks) != 400:
        raise RuntimeError(f"expected 400 tasks, found {len(tasks)}")
    dev_templates = {task["template_id"] for task in tasks if task["split"] == "dev"}
    test_templates = {task["template_id"] for task in tasks if task["split"] == "test"}
    if not dev_templates or not test_templates:
        raise RuntimeError("dev and test splits must both be non-empty")
    if not dev_templates.isdisjoint(test_templates):
        raise RuntimeError("dev and test templates overlap")
    dev_ids = {task["task_id"] for task in tasks if task["split"] == "dev"}
    test_ids = {task["task_id"] for task in tasks if task["split"] == "test"}
    if not dev_ids.isdisjoint(test_ids):
        raise RuntimeError("dev and test task ids overlap")
    if any(not 1 <= task["n_steps"] <= 6 for task in tasks):
        raise RuntimeError("tasks must use 1 to 6 tool calls")
    counts = {
        "n_tools": n_tools,
        "n_tasks": len(tasks),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "n_dev_templates": len(dev_templates),
        "n_test_templates": len(test_templates),
    }
    if counts["n_dev_templates"] != 40 or counts["n_test_templates"] != 40:
        raise RuntimeError("expected 40 dev templates and 40 test templates")
    return counts


def save_dataset(data_dir: Path, tools: list[ToolSpec], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the generated dataset once. Later runs must reproduce the same bytes."""
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = verify_task_split(tasks, len(tools))
    tool_payload = [tool.model_dump() for tool in tools]
    tools_bytes = canonical_bytes(tool_payload)
    tasks_bytes = canonical_bytes(tasks)
    manifest = {
        "seed": 0,
        "license": "generated locally by this project",
        **counts,
        "tools_sha256": sha256_bytes(tools_bytes),
        "tasks_sha256": sha256_bytes(tasks_bytes),
    }
    tools_path = data_dir / "tools.json"
    tasks_path = data_dir / "tasks.json"
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("tools_sha256") != manifest["tools_sha256"] or existing.get("tasks_sha256") != manifest["tasks_sha256"]:
            raise RuntimeError(
                "Saved dev/test split does not match the generator. "
                "Refusing to regenerate. Delete data/ only if you intend to rebuild the split."
            )
        _check_file(tools_path, manifest["tools_sha256"])
        _check_file(tasks_path, manifest["tasks_sha256"])
        return existing
    tools_path.write_bytes(tools_bytes)
    tasks_path.write_bytes(tasks_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _check_file(tools_path, manifest["tools_sha256"])
    _check_file(tasks_path, manifest["tasks_sha256"])
    reloaded = json.loads(tasks_path.read_text(encoding="utf-8"))
    verify_task_split(reloaded, len(json.loads(tools_path.read_text(encoding="utf-8"))))
    return manifest


def _check_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"checksum mismatch for {path.name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified_stream(url: str, dest: Path, expected_sha256: str, session: Any | None = None) -> Path:
    """Stream a large file to disk and verify sha256. Cached files are reused."""
    if dest.exists() and sha256_file(dest) == expected_sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    client = session if session is not None else requests
    response = client.get(url, stream=True, timeout=120)
    response.raise_for_status()
    digest = hashlib.sha256()
    written = 0
    mark = 64 * 1024 * 1024
    print(f"download start {dest.name}", flush=True)
    try:
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                previous = written
                written += len(chunk)
                if written // mark != previous // mark:
                    print(f"download {dest.name} {written // (1024 * 1024)} MiB", flush=True)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"checksum mismatch for {url}: {digest.hexdigest()}")
        if dest.exists():
            dest.unlink()
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()
    return dest


def download_verified(url: str, dest: Path, expected_sha256: str, session: Any | None = None) -> Path:
    """Download with requests and verify sha256. Cached files are reused."""
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected_sha256:
        return dest
    client = session if session is not None else requests
    response = client.get(url, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {url}: {digest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def ensure_minilm_config(data_dir: Path) -> Path:
    dest = data_dir / "models" / "all-MiniLM-L6-v2" / "config.json"
    return download_verified(MINILM_CONFIG_URL, dest, MINILM_CONFIG_SHA256)
