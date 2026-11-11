"""Offline tests: a scripted fake LLM drives the real agent loop and tools."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent  # noqa: E402
from agent.llm import parse_response  # noqa: E402
from agent.tools import TOOLS, ToolContext, ToolError, safe_path  # noqa: E402


class FakeLLM:
    """Returns pre-written JSON replies in order, validating them like the real client."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def next_step(self, history):
        self.seen.append(list(history))
        raw = json.dumps(self.replies.pop(0))
        return parse_response(raw), raw


def make_agent(tmp_path, replies):
    llm = FakeLLM(replies)
    agent = Agent(tmp_path, llm=llm, session_id="test")
    events = []
    agent.emit = events.append
    return agent, llm, events


def test_parse_handles_fences_and_rejects_unknown_tool():
    ok = parse_response('```json\n{"thought": "x", "actions": [], "final_answer": "hi"}\n```')
    assert ok.done and ok.final_answer == "hi"
    with pytest.raises(ValueError, match="unknown tool"):
        parse_response('{"thought": "x", "actions": [{"tool": "rm_rf", "args": {}}]}')
    with pytest.raises(ValueError, match="invalid args"):
        parse_response('{"thought": "x", "actions": [{"tool": "read_file", "args": {}}]}')
    with pytest.raises(ValueError):
        parse_response("Sure! I'll do that.")


def test_path_traversal_blocked(tmp_path):
    ctx = ToolContext(tmp_path)
    with pytest.raises(ToolError):
        safe_path(ctx, "../secret.txt")
    with pytest.raises(ToolError):
        safe_path(ctx, "C:/Windows/system.ini" if sys.platform == "win32" else "/etc/passwd")


def test_calculate_is_safe(tmp_path):
    calc = TOOLS["calculate"]
    ctx = ToolContext(tmp_path)
    assert calc.func(ctx, calc.args_model(expression="2340 * 0.15"))["result"] == pytest.approx(351)
    with pytest.raises(ToolError):
        calc.func(ctx, calc.args_model(expression="__import__('os').system('dir')"))


def test_multi_step_write_then_answer(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        {"thought": "write it", "actions": [
            {"tool": "write_file", "args": {"path": "a.txt", "content": "hello"}}], "final_answer": None},
        {"thought": "done", "actions": [], "final_answer": "Saved a.txt"},
    ])
    agent.submit("save hello to a.txt")
    assert (tmp_path / "a.txt").read_text() == "hello"
    assert events[-1].kind == "final" and "a.txt" in events[-1].text
    # the observation was fed back to the LLM
    assert "OBSERVATIONS" in llm.seen[1][-1]["content"]


def test_destructive_action_waits_and_can_be_denied(tmp_path):
    (tmp_path / "keep.txt").write_text("important")
    agent, llm, events = make_agent(tmp_path, [
        {"thought": "delete", "actions": [{"tool": "delete_path", "args": {"path": "keep.txt"}}],
         "final_answer": None},
        {"thought": "user said no", "actions": [], "final_answer": "Okay, I kept it."},
    ])
    agent.submit("delete keep.txt")
    assert agent.awaiting is not None and (tmp_path / "keep.txt").exists()
    agent.resolve(False)
    assert (tmp_path / "keep.txt").exists()
    assert agent.awaiting is None and events[-1].kind == "final"


def test_destructive_action_runs_after_approval(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    agent, _, _ = make_agent(tmp_path, [
        {"thought": "delete", "actions": [{"tool": "delete_path", "args": {"path": "old.txt"}}],
         "final_answer": None},
        {"thought": "ok", "actions": [], "final_answer": "Deleted."},
    ])
    agent.submit("delete old.txt")
    agent.resolve(True)
    assert not (tmp_path / "old.txt").exists()


def test_task_database_roundtrip(tmp_path):
    agent, _, _ = make_agent(tmp_path, [
        {"thought": "add", "actions": [
            {"tool": "add_task", "args": {"title": "Record demo", "priority": "high"}},
            {"tool": "add_task", "args": {"title": "Write README"}}], "final_answer": None},
        {"thought": "complete", "actions": [{"tool": "complete_task", "args": {"id": 1}}],
         "final_answer": None},
        {"thought": "done", "actions": [], "final_answer": "ok"},
    ])
    agent.submit("add tasks")
    ctx = ToolContext(tmp_path)
    lt = TOOLS["list_tasks"]
    open_tasks = lt.func(ctx, lt.args_model(status="open"))["tasks"]
    assert [t["title"] for t in open_tasks] == ["Write README"]


def test_tool_errors_are_reported_not_raised(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        {"thought": "read", "actions": [{"tool": "read_file", "args": {"path": "nope.txt"}}],
         "final_answer": None},
        {"thought": "missing", "actions": [], "final_answer": "That file doesn't exist."},
    ])
    agent.submit("read nope.txt")
    failed = [e for e in events if e.kind == "action"][0]
    assert failed.data["ok"] is False
    assert "does not exist" in llm.seen[1][-1]["content"]
