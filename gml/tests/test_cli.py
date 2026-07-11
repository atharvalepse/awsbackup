"""CLI smoke test — runs `gml ask --stub-client` as a subprocess."""
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_cli_ask_with_stub_client(tmp_path):
    """End-to-end CLI invocation against a StubClient + isolated memory file."""
    env_memory = tmp_path / "memories.jsonl"
    result = subprocess.run(
        [
            sys.executable, "-m", "orchestration.cli",
            "ask", "hello world",
            "--target", "deepseek",
            "--stub-client",
            "--no-extract",
            "--no-sam-llm",
            "--embedder", "stub",  # explicit so the test doesn't need Ollama/Gemini
            "--memory-path", str(env_memory),
        ],
        capture_output=True,
        text=True,
        cwd=_project_root(),
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "stub response" in result.stdout


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "orchestration.cli", "--version"],
        capture_output=True, text=True, cwd=_project_root(), timeout=10,
    )
    assert result.returncode == 0
    assert "gml" in result.stdout
