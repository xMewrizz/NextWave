"""Pinned local GGUF runtime for hackathon inference without a cloud API key."""

from __future__ import annotations

import atexit
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOCAL_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_MODEL_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
LOCAL_MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/"
    f"{LOCAL_MODEL_FILENAME}?download=true"
)
LOCAL_MODEL_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"
LOCAL_SERVER_ARCHIVE = "llama-b11081-bin-win-cpu-x64.zip"
LOCAL_SERVER_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b11081/"
    f"{LOCAL_SERVER_ARCHIVE}"
)
LOCAL_SERVER_SHA256 = "48f13c153946cca8543fd3ab915709ec5f340bfe1f38c687121b3d58f848b7b2"
LOCAL_SERVER_ROOT = "http://127.0.0.1:18080"

_START_LOCK = threading.Lock()
_SERVER_PROCESS: subprocess.Popen[bytes] | None = None
_VERIFIED_MODEL_PATH: Path | None = None


def _runtime_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "runtime" / "llm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, path: Path, sha256: str) -> None:
    if path.is_file() and _sha256(path) == sha256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "NextWave/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            append = offset > 0 and response.status == 206
            with partial.open("ab" if append else "wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
    except (HTTPError, URLError, OSError) as error:
        raise RuntimeError(f"Local LLM download failed: {path.name}") from error
    if _sha256(partial) != sha256:
        raise RuntimeError(f"Local LLM checksum mismatch: {path.name}")
    os.replace(partial, path)


def _server_executable(directory: Path) -> Path:
    if sys.platform == "win32" and platform.machine().casefold() in {"amd64", "x86_64"}:
        executable = directory / "bin" / "llama-server.exe"
        if executable.is_file():
            return executable
        archive = directory / LOCAL_SERVER_ARCHIVE
        _download_verified(LOCAL_SERVER_URL, archive, LOCAL_SERVER_SHA256)
        binary_dir = directory / "bin"
        binary_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename)
                if len(name.parts) != 1:
                    raise RuntimeError("Unexpected path in local LLM runtime archive")
                with package.open(member) as source, (binary_dir / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
        if not executable.is_file():
            raise RuntimeError("Local LLM runtime archive has no llama-server.exe")
        return executable
    system_executable = shutil.which("llama-server")
    if system_executable is None:
        raise RuntimeError("Install llama.cpp's llama-server for this operating system")
    return Path(system_executable)


def _healthy() -> bool:
    try:
        with urlopen(f"{LOCAL_SERVER_ROOT}/health", timeout=2) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _stop_server() -> None:
    if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
        _SERVER_PROCESS.terminate()


atexit.register(_stop_server)


def ensure_local_llm_server() -> str:
    """Download pinned assets on first use and start a loopback-only server."""
    global _SERVER_PROCESS, _VERIFIED_MODEL_PATH

    with _START_LOCK:
        if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None and _healthy():
            return LOCAL_SERVER_ROOT
        directory = _runtime_directory()
        model = directory / LOCAL_MODEL_FILENAME
        if _VERIFIED_MODEL_PATH != model:
            _download_verified(LOCAL_MODEL_URL, model, LOCAL_MODEL_SHA256)
            _VERIFIED_MODEL_PATH = model
        executable = _server_executable(directory)
        if _healthy():
            raise RuntimeError("Local LLM port 18080 is already in use by another server")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        _SERVER_PROCESS = subprocess.Popen(
            [
                str(executable), "-m", str(model), "-c", "8192", "-t", "8",
                "-np", "1", "--host", "127.0.0.1", "--port", "18080", "--no-webui",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if _SERVER_PROCESS.poll() is not None:
                raise RuntimeError("Local LLM server exited during startup")
            if _healthy():
                return LOCAL_SERVER_ROOT
            time.sleep(0.5)
        _stop_server()
        raise RuntimeError("Local LLM server did not become ready within 120 seconds")
