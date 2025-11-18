"""
Unit tests for the Ollama orchestration helper.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from core.llm.local_llm import (
    DEFAULT_MODEL_ID,
    OllamaSettings,
    _run_command,
    setup_ollama_llm,
)


class TestRunCommand:
    """Tests for the _run_command helper function."""

    @patch("core.llm.local_llm.subprocess.run")
    def test_successful_command(self, mock_run):
        """Test that successful commands return the CompletedProcess."""
        mock_result = MagicMock()
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = _run_command("echo test")

        assert result.stdout == "success output"
        mock_run.assert_called_once_with(
            "echo test",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=None,
        )

    @patch("core.llm.local_llm.subprocess.run")
    def test_failed_command_raises_runtime_error(self, mock_run):
        """Test that failed commands raise RuntimeError with context."""
        error = subprocess.CalledProcessError(1, "bad command")
        error.stdout = "some output"
        error.stderr = "error details"
        mock_run.side_effect = error

        with pytest.raises(RuntimeError, match="Command failed: bad command"):
            _run_command("bad command")

    @patch("core.llm.local_llm.subprocess.run")
    def test_timeout_raises_runtime_error(self, mock_run):
        """Test that timeouts raise RuntimeError with context."""
        error = subprocess.TimeoutExpired("slow command", 10)
        error.stdout = b"partial output"
        error.stderr = b"timeout error"
        mock_run.side_effect = error

        with pytest.raises(
            RuntimeError, match="Command timed out after 10 seconds: slow command"
        ):
            _run_command("slow command", timeout=10)


class TestSetupOllamaLLM:
    """Tests for the public setup_ollama_llm function."""

    @patch("core.llm.local_llm.requests.get")
    @patch("core.llm.local_llm.subprocess.Popen")
    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value="/usr/bin/ollama")
    def test_pulls_model_when_missing(
        self, mock_which, mock_run_command, mock_popen, mock_get
    ):
        """Test that setup pulls a model when it's not in the list."""
        # Mock: ollama list returns output without our model
        list_result = MagicMock()
        list_result.stdout = "NAME\tTAG\tSIZE\nsome-other-model\tlatest\t4.0 GB"

        # Mock: ollama pull succeeds
        pull_result = MagicMock()
        pull_result.stdout = "success"

        mock_run_command.side_effect = [list_result, pull_result]

        # Mock: server is already running
        mock_get.return_value = MagicMock()

        model_id = setup_ollama_llm()

        assert model_id == DEFAULT_MODEL_ID
        assert mock_run_command.call_count == 2
        # Verify ollama pull was called
        assert "ollama pull" in str(mock_run_command.call_args_list[1])

    @patch("core.llm.local_llm.requests.get")
    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value="/usr/bin/ollama")
    def test_skips_pull_when_model_present(
        self, mock_which, mock_run_command, mock_get
    ):
        """Test that setup skips pulling when the model is already present."""
        # Mock: ollama list returns output with our model
        list_result = MagicMock()
        list_result.stdout = f"NAME\tTAG\tSIZE\n{DEFAULT_MODEL_ID}\tlatest\t4.0 GB"
        mock_run_command.return_value = list_result

        # Mock: server is already running
        mock_get.return_value = MagicMock()

        setup_ollama_llm()

        # Should only call 'ollama list', not 'ollama pull'
        assert mock_run_command.call_count == 1

    @patch("core.llm.local_llm.time.sleep")
    @patch("core.llm.local_llm.time.time")
    @patch("core.llm.local_llm.requests.get")
    @patch("core.llm.local_llm.subprocess.Popen")
    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value="/usr/bin/ollama")
    def test_starts_server_when_not_running(
        self, mock_which, mock_run_command, mock_popen, mock_get, mock_time, mock_sleep
    ):
        """Test that setup starts the Ollama server when it's not running."""
        # Mock: ollama list returns model present
        list_result = MagicMock()
        list_result.stdout = f"NAME\tTAG\tSIZE\n{DEFAULT_MODEL_ID}\tlatest\t4.0 GB"
        mock_run_command.return_value = list_result

        # Mock: server not running initially, then starts
        mock_get.side_effect = [
            requests.RequestException("Server not available"),
            MagicMock(),  # Server responds after starting
        ]

        # Mock time to prevent infinite loop
        mock_time.side_effect = [0, 5]  # First check at t=0, second at t=5

        setup_ollama_llm()

        # Verify server was started
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == ["ollama", "serve"]

    @patch("core.llm.local_llm.time.sleep")
    @patch("core.llm.local_llm.time.time")
    @patch("core.llm.local_llm.requests.get")
    @patch("core.llm.local_llm.subprocess.Popen")
    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value="/usr/bin/ollama")
    def test_raises_when_server_fails_to_start(
        self, mock_which, mock_run_command, mock_popen, mock_get, mock_time, mock_sleep
    ):
        """Test that setup raises RuntimeError when server won't start."""
        # Mock: ollama list returns model present
        list_result = MagicMock()
        list_result.stdout = f"NAME\tTAG\tSIZE\n{DEFAULT_MODEL_ID}\tlatest\t4.0 GB"
        mock_run_command.return_value = list_result

        # Mock: server never becomes available
        mock_get.side_effect = requests.RequestException("Server not available")

        # Mock time to simulate timeout
        mock_time.side_effect = [0, 61]  # Exceeds default 60-second timeout

        with pytest.raises(
            RuntimeError, match="Ollama server failed to start.*within 60 seconds"
        ):
            setup_ollama_llm()

    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value=None)
    def test_installs_ollama_when_not_present(self, mock_which, mock_run_command):
        """Test that setup installs Ollama when the CLI is not found."""
        # Mock: installation succeeds
        install_result = MagicMock()
        install_result.stdout = "Ollama installed"

        # Mock: ollama list after installation
        list_result = MagicMock()
        list_result.stdout = f"NAME\tTAG\tSIZE\n{DEFAULT_MODEL_ID}\tlatest\t4.0 GB"

        mock_run_command.side_effect = [install_result, list_result]

        # Mock: We need to patch shutil.which to return a path after installation
        with patch(
            "core.llm.local_llm.shutil.which", side_effect=[None, "/usr/bin/ollama"]
        ):
            with patch("core.llm.local_llm.requests.get"):
                setup_ollama_llm()

        # Verify installation command was called
        assert "curl -fsSL https://ollama.com/install.sh | sh" in str(
            mock_run_command.call_args_list[0]
        )

    @patch("core.llm.local_llm.requests.get")
    @patch("core.llm.local_llm._run_command")
    @patch("core.llm.local_llm.shutil.which", return_value="/usr/bin/ollama")
    def test_accepts_custom_settings(self, mock_which, mock_run_command, mock_get):
        """Test that setup respects custom OllamaSettings."""
        custom_model = "custom-model"
        settings = OllamaSettings(model_id=custom_model)

        # Mock: ollama list returns custom model
        list_result = MagicMock()
        list_result.stdout = f"NAME\tTAG\tSIZE\n{custom_model}\tlatest\t2.0 GB"
        mock_run_command.return_value = list_result

        # Mock: server is running
        mock_get.return_value = MagicMock()

        returned_model = setup_ollama_llm(settings=settings)

        assert returned_model == custom_model
