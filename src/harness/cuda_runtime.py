"""Windows CUDA 12 runtime DLLs for the llama.cpp wheel.

The cu124 wheel links cublas64_12.dll and cudart64_12.dll, which are not part of
the Python package. They are downloaded from NVIDIA's CUDA redistributable,
checked by sha256, and placed on the DLL search path before llama.cpp loads.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from evals.data import download_verified_stream

# CUDA 12.4.1 redistributable entries (windows-x86_64) from redistrib_12.4.1.json.
CUDA_PACKAGES: tuple[tuple[str, str, str], ...] = (
    (
        "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip",
        "6a1c32e68ee1a95ca17334691ff9ad1ffe7f352c24a083d55e4c96b8063b2bcb",
        "cuda_cudart-windows-x86_64-12.4.127-archive.zip",
    ),
    (
        "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip",
        "698140f12da055a3709eee2e022fcfe7bc8edf31f30115e3f7a5c877a9491de5",
        "libcublas-windows-x86_64-12.4.5.8-archive.zip",
    ),
)


def prepare_windows_cuda_runtime(data_dir: Path) -> Path | None:
    """Return the directory of extracted CUDA DLLs, or None off Windows."""
    if os.name != "nt":
        return None
    bin_dir = data_dir / "cuda" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if not (bin_dir / "cudart64_12.dll").exists() or not (bin_dir / "cublas64_12.dll").exists():
        cache = data_dir / "cuda"
        for url, digest, filename in CUDA_PACKAGES:
            archive = download_verified_stream(url, cache / filename, digest)
            _extract_dlls(archive, bin_dir)
    _register(bin_dir)
    return bin_dir


def _extract_dlls(archive: Path, bin_dir: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            name = Path(info.filename).name
            if not name.lower().endswith(".dll"):
                continue
            if "/" in info.filename.replace("\\", "/").rstrip("/") and info.is_dir():
                continue
            target = bin_dir / name
            if target.exists():
                continue
            target.write_bytes(bundle.read(info))


def _register(bin_dir: Path) -> None:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir.resolve()))
    os.environ["PATH"] = str(bin_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
