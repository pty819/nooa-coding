from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from nooa_coding.config import (
    CodingSettings,
    CompactionSettings,
    MemorySettings,
    PermissionSettings,
)


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("# fixture\n", encoding="utf-8")
    run_git(repo, "add", "README.md")
    run_git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initial",
    )
    return repo


@pytest.fixture
def settings(tmp_path: Path) -> CodingSettings:
    return CodingSettings(
        sessions_dir=str(tmp_path / "state" / "sessions"),
        worktrees_dir=str(tmp_path / "state" / "worktrees"),
        verification_commands=("python3 -c 'print(\"verified\")'",),
        permissions=PermissionSettings(file_write="allow", shell="allow"),
        memory=MemorySettings(enabled=False),
        compaction=CompactionSettings(enabled=False),
    )


def response_with_code(code: str) -> LLMResponse:
    call = ToolCall(
        id="execute",
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[call],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


def coding_response(*, write_file: bool = True, status: str = "completed") -> LLMResponse:
    write = "await self.shell.write_file('result.txt', 'done\\n')\n" if write_file else ""
    code = f"""{write}return_result(CodingTaskDraft(
    status={status!r},
    summary='implemented fixture change',
    root_cause='missing fixture output',
    changed_files=['result.txt'] if {write_file!r} else [],
    evidence='observed scripted tool output',
    suggested_verification='python3 -c \\\"print(1)\\\"',
))"""
    return response_with_code(code)


def fake_llm(*responses: LLMResponse) -> FakeLLMClient:
    return FakeLLMClient(scripted_responses=list(responses))


__all__ = ["coding_response", "fake_llm", "response_with_code", "run_git"]
