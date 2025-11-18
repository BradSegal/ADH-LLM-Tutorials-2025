"""
Ollama orchestration for local LLM runtime.

This module configures and manages a local Ollama server, providing OpenAI-compatible
API access to local LLMs for use in educational notebooks.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Final

import requests
from pydantic import BaseModel, Field

# Use a model that supports tool calling and structured output.
DEFAULT_MODEL_ID: Final[str] = "llama3.2:3b-instruct-q4_K_M"
_HEALTH_ENDPOINT: Final[str] = "/api/version"


class OllamaSettings(BaseModel):
    """Configuration for the local Ollama runtime."""

    base_url: str = Field(default="http://127.0.0.1:11434")
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    pull_timeout: int = Field(default=600)  # 10 minutes
    health_timeout: int = Field(default=60)  # 1 minute


def _normalize_stream(stream: bytes | str | None) -> str:
    """Return a human readable representation of a completed process stream."""
    if stream is None:
        return "N/A"
    if isinstance(stream, bytes):
        return stream.decode(errors="replace").strip()
    return stream.strip()


def _run_command(
    command: str, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    """Runs a shell command, failing loudly with informative errors."""
    try:
        return subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Command failed: {command}\n"
            f"STDOUT: {e.stdout.strip()}\n"
            f"STDERR: {e.stderr.strip()}"
        )
        raise RuntimeError(error_message) from e
    except subprocess.TimeoutExpired as e:
        stdout_str = _normalize_stream(e.stdout)
        stderr_str = _normalize_stream(e.stderr)
        error_message = (
            f"Command timed out after {timeout} seconds: {command}\n"
            f"STDOUT: {stdout_str}\n"
            f"STDERR: {stderr_str}"
        )
        raise RuntimeError(error_message) from e


def _healthcheck_url(base_url: str) -> str:
    """Return the Ollama health endpoint for the provided base URL."""
    base = base_url.rstrip("/")
    return f"{base}{_HEALTH_ENDPOINT}"


def _wait_for_server(base_url: str, timeout: int) -> None:
    """Wait for the Ollama HTTP endpoint to become responsive."""
    deadline = time.time() + timeout
    health_url = _healthcheck_url(base_url)
    while time.time() < deadline:
        try:
            requests.get(health_url, timeout=2).raise_for_status()
            print("✅ Ollama server started successfully.")
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(
        f"Ollama server failed to start at {base_url} within {timeout} seconds."
    )


def setup_ollama_llm(settings: OllamaSettings | None = None) -> str:
    """
    Ensures Ollama is installed, the server is running, and the model is pulled.

    This function is idempotent and designed for a Colab/Linux environment.

    Parameters
    ----------
    settings : OllamaSettings, optional
        Configuration for the Ollama server and model. If not provided,
        defaults will be used.

    Returns
    -------
    str
        The model ID to be used in the OpenAI client.

    Raises
    ------
    RuntimeError
        If Ollama installation fails, server cannot start, or model pull fails.
    """
    cfg = settings or OllamaSettings()

    # 1. Install Ollama if not present
    if shutil.which("ollama") is None:
        print("Ollama CLI not found. Installing now (this is a one-time setup)...")
        _run_command("curl -fsSL https://ollama.com/install.sh | sh")
        print("✅ Ollama installed successfully.")

    # 2. Start the Ollama server as a background daemon if not running
    health_url = _healthcheck_url(cfg.base_url)
    try:
        requests.get(health_url, timeout=2).raise_for_status()
        print("✅ Ollama server is already running.")
    except requests.RequestException:
        print("Starting Ollama server in the background...")
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        _wait_for_server(cfg.base_url, cfg.health_timeout)

    # 3. Pull the required model if not already available
    result = _run_command("ollama list")
    if cfg.model_id not in result.stdout:
        print(f"📥 Pulling model '{cfg.model_id}'. This may take several minutes...")
        _run_command(f"ollama pull {cfg.model_id}", timeout=cfg.pull_timeout)
        print(f"✅ Model '{cfg.model_id}' pulled successfully.")
    else:
        print(f"✅ Model '{cfg.model_id}' is already available.")

    return cfg.model_id
