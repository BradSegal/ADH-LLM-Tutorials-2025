"""
Live integration tests for the Ollama helper and OpenAI-compatible workflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest
import requests
from openai import OpenAI
from pydantic import BaseModel

from core.llm import OllamaSettings, setup_ollama_llm
from core.llm.tools import calculate_bmi, get_drug_interaction

pytestmark = pytest.mark.api

_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_OLLAMA_API_BASE = f"{_OLLAMA_BASE_URL}/v1"


def _ensure_user_bins_on_path() -> None:
    """Ensure user-level install locations are on PATH for the test session."""
    path = os.environ.get("PATH", "")
    candidate_dirs = [
        Path.home() / "bin",
        Path.home() / ".ollama" / "bin",
    ]
    for directory in candidate_dirs:
        if directory.exists():
            dir_str = str(directory)
            if dir_str not in path:
                path = f"{dir_str}:{path}"
    os.environ["PATH"] = path


def _terminate_ollama_processes() -> None:
    """Kill any running ollama processes to guarantee a clean slate."""
    for process in psutil.process_iter(["name"]):
        name = process.info.get("name") or ""
        if "ollama" not in name.lower():
            continue
        try:
            process.kill()
            process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue


def _wait_for_http(timeout: int = 60) -> None:
    """Wait for the Ollama HTTP API to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{_OLLAMA_BASE_URL}/api/version", timeout=2)
            resp.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("Ollama server failed to start during test setup.")


@pytest.fixture(scope="module", autouse=True)
def ollama_server() -> Iterator[None]:
    """Start the Ollama server for the integration test module."""
    _ensure_user_bins_on_path()
    _terminate_ollama_processes()
    server_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_http()
        yield
    finally:
        _terminate_ollama_processes()
        # Fall back to terminating the server process if it is still alive.
        if server_process.poll() is None:
            server_process.kill()


@pytest.fixture(scope="module")
def ollama_test_settings() -> OllamaSettings:
    """Provide configurable settings for integration tests."""
    default_settings = OllamaSettings()
    model_id = os.environ.get("OLLAMA_TEST_MODEL_ID", default_settings.model_id)
    pull_timeout = int(os.environ.get("OLLAMA_PULL_TIMEOUT", "900"))
    health_timeout = int(os.environ.get("OLLAMA_HEALTH_TIMEOUT", "60"))
    return OllamaSettings(
        base_url=default_settings.base_url,
        model_id=model_id,
        pull_timeout=pull_timeout,
        health_timeout=health_timeout,
    )


@pytest.fixture(scope="module")
def openai_context(
    ollama_server: None, ollama_test_settings: OllamaSettings
) -> tuple[OpenAI, str]:
    """Ensure the helper runs once and provide a configured OpenAI client."""
    model_name = setup_ollama_llm(ollama_test_settings)
    client = OpenAI(base_url=_OLLAMA_API_BASE, api_key="ollama")
    return client, model_name


def _remove_model(model_id: str) -> None:
    """Remove an Ollama model if present."""
    subprocess.run(
        ["ollama", "rm", model_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def test_setup_ollama_llm_pulls_model(ollama_test_settings: OllamaSettings) -> None:
    """Ensure the helper downloads the model when it is missing."""
    _remove_model(ollama_test_settings.model_id)

    model_id = setup_ollama_llm(ollama_test_settings)
    assert model_id == ollama_test_settings.model_id

    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert ollama_test_settings.model_id in result.stdout


def test_setup_ollama_llm_is_idempotent(ollama_test_settings: OllamaSettings) -> None:
    """Confirm the helper short-circuits when the model is already cached."""
    max_seconds = float(os.environ.get("OLLAMA_IDEMPOTENT_MAX_SECONDS", "20"))

    # Ensure the model is already available from the previous test.
    setup_ollama_llm(ollama_test_settings)

    start = time.time()
    setup_ollama_llm(ollama_test_settings)
    elapsed = time.time() - start

    assert (
        elapsed < max_seconds
    ), f"Expected idempotent setup in <{max_seconds}s, took {elapsed:.2f}s."


def test_basic_completion_with_openai_client(
    openai_context: tuple[OpenAI, str],
) -> None:
    """Smoke test basic chat completion through the OpenAI client."""
    client, model_name = openai_context
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Reply with a short greeting."}],
        max_tokens=64,
    )
    content = response.choices[0].message.content
    assert content is not None
    assert content.strip() != ""


class PatientSummary(BaseModel):
    """Structured representation of a patient summary."""

    patient_id: str
    age: int
    diagnosis: str


def test_structured_output_with_openai_client(
    openai_context: tuple[OpenAI, str],
) -> None:
    """Validate that structured parsing works end-to-end."""
    client, model_name = openai_context
    clinical_note = (
        "Patient ID: Q3321\nAge: 48\nPrimary Diagnosis: Hypertension.\n"
        "Return JSON with those fields."
    )

    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "Extract fields accurately."},
            {"role": "user", "content": clinical_note},
        ],
        response_format=PatientSummary,
    )

    parsed = response.choices[0].message.parsed
    assert parsed is not None
    assert parsed.patient_id == "Q3321"
    assert parsed.age == 48
    assert "hypertension" in parsed.diagnosis.lower()


def test_tool_call_flow_with_openai_client(openai_context: tuple[OpenAI, str]) -> None:
    """Run the two-step tool execution workflow via the OpenAI client."""
    client, model_name = openai_context
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_bmi",
                "description": "Compute BMI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight_kg": {"type": "number"},
                        "height_m": {"type": "number"},
                    },
                    "required": ["weight_kg", "height_m"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_drug_interaction",
                "description": "Check for interactions between two drugs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_a": {"type": "string"},
                        "drug_b": {"type": "string"},
                    },
                    "required": ["drug_a", "drug_b"],
                },
            },
        },
    ]

    initial_messages = [
        {
            "role": "system",
            "content": "You are a clinical assistant with access to tools.",
        },
        {
            "role": "user",
            "content": "A patient weighs 87 kg and is 1.82m tall. "
            "Use the BMI tool and include the numeric result.",
        },
    ]

    first_response = client.chat.completions.create(
        model=model_name,
        messages=initial_messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=256,
    )
    assistant_message = first_response.choices[0].message

    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        function_name = tool_call.function.name
        numeric_result: float | None = None
        tool_result: float | int | str
        if function_name == "calculate_bmi":
            tool_result = calculate_bmi(
                weight_kg=float(arguments["weight_kg"]),
                height_m=float(arguments["height_m"]),
            )
            numeric_result = float(tool_result)
        elif function_name == "get_drug_interaction":
            tool_result = get_drug_interaction(
                drug_a=str(arguments["drug_a"]),
                drug_b=str(arguments["drug_b"]),
            )
        else:  # pragma: no cover - defensive: unexpected tool call
            raise AssertionError(f"Unexpected tool requested: {function_name}")
        assert isinstance(tool_result, (float, int, str))

        follow_up_messages = initial_messages + [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(tool_result),
            },
        ]
        final_response = client.chat.completions.create(
            model=model_name,
            messages=follow_up_messages,
            tools=tools,
            max_tokens=256,
        )

        final_content = final_response.choices[0].message.content or ""
        if numeric_result is not None:
            assert str(round(numeric_result, 2)) in final_content
    else:
        # Some models may skip tool calls and provide an answer directly.
        assert assistant_message.content is not None
