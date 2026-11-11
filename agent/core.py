"""The agent loop: LLM plan -> validate -> (confirm) -> execute -> observe -> repeat.

Written as a small state machine so that both the terminal UI and the web UI can
pause for a confirmation on destructive actions and then resume.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .audit import log_event
from .llm import LLMClient, LLMError
from .schema import Action
from .tools import TOOLS, ToolContext, ToolError

MAX_STEPS = 6            # LLM turns per user request (prevents runaway loops)
MAX_OBS_CHARS = 6000     # keep observations small so context stays cheap


@dataclass
class Event:
    """Something the UI should show. kind: user|thought|action|final|error|confirm|info"""
    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(self, workspace: Path, api_key: str | None = None,
                 session_id: str | None = None, llm: Optional[LLMClient] = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.ctx = ToolContext(workspace=workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        self.llm = llm or LLMClient(api_key=api_key, on_event=self._log)
        self.history: list[dict] = []
        self.pending: list[Action] = []
        self.observations: list[dict] = []
        self.steps = 0
        self.awaiting: Optional[Action] = None
        self._approved_next = False
        self.emit: Callable[[Event], None] = lambda e: None

    # ------------------------------------------------------------ public API

    @property
    def busy(self) -> bool:
        return self.awaiting is not None

    def submit(self, text: str) -> None:
        if self.busy:
            raise RuntimeError("Answer the pending confirmation first.")
        text = text.strip()
        if not text:
            return
        self._log("user_request", text=text)
        self.history.append({"role": "user", "content": text})
        self.steps = 0
        self.pending, self.observations = [], []
        self._run()

    def resolve(self, approved: bool) -> None:
        """Answer a pending confirmation, then continue the loop."""
        action = self.awaiting
        if action is None:
            return
        self.awaiting = None
        self._log("confirmation", tool=action.tool, args=action.args, approved=approved)
        if approved:
            self._approved_next = True
        else:
            self.pending.pop(0)
            self._observe(action, ok=False, error="User denied this action. Do not retry it.")
            self.emit(Event("action", f"Skipped `{action.tool}` (denied by you)",
                            {"tool": action.tool, "args": action.args, "ok": False, "denied": True}))
        self._run()

    # ------------------------------------------------------------ loop

    def _run(self) -> None:
        try:
            while True:
                if not self._execute_pending():
                    return  # paused for confirmation
                if self.observations:
                    self.history.append({
                        "role": "user",
                        "content": "OBSERVATIONS: " + json.dumps(self.observations, default=str),
                    })
                    self.observations = []
                if self.steps >= MAX_STEPS:
                    msg = f"Stopped after {MAX_STEPS} steps to stay safe. Ask me to continue if needed."
                    self.history.append({"role": "assistant", "content": json.dumps(
                        {"thought": "step limit", "actions": [], "final_answer": msg})})
                    self.emit(Event("final", msg))
                    return
                resp, raw = self.llm.next_step(self.history)
                self.steps += 1
                self.history.append({"role": "assistant", "content": raw})
                self._log("plan", thought=resp.thought,
                          actions=[a.model_dump() for a in resp.actions], final=resp.final_answer)
                self.emit(Event("thought", resp.thought))
                if resp.done or not resp.actions:
                    self.emit(Event("final", resp.final_answer or "Done."))
                    return
                self.pending = list(resp.actions)
        except LLMError as e:
            self._log("error", error=str(e))
            self.emit(Event("error", str(e)))
            self.pending, self.observations = [], []

    def _execute_pending(self) -> bool:
        """Run queued actions. Returns False if paused waiting for the user."""
        while self.pending:
            action = self.pending[0]
            tool = TOOLS[action.tool]  # existence already validated in llm.parse_response
            try:
                args = tool.args_model.model_validate(action.args)
            except ValidationError as e:
                self.pending.pop(0)
                self._observe(action, ok=False, error=f"invalid args: {e.errors(include_url=False)}")
                continue

            if tool.is_destructive(self.ctx, args) and not self._approved_next:
                self.awaiting = action
                self._log("confirmation_requested", tool=action.tool, args=action.args)
                self.emit(Event("confirm", f"`{action.tool}` needs your approval",
                                {"tool": action.tool, "args": action.args, "reason": action.reason}))
                return False
            self._approved_next = False
            self.pending.pop(0)

            try:
                result = tool.func(self.ctx, args)
                self._observe(action, ok=True, result=result)
                self.emit(Event("action", f"Ran `{action.tool}`",
                                {"tool": action.tool, "args": action.args, "ok": True,
                                 "result": result, "reason": action.reason}))
            except ToolError as e:
                self._observe(action, ok=False, error=str(e))
                self.emit(Event("action", f"`{action.tool}` failed: {e}",
                                {"tool": action.tool, "args": action.args, "ok": False, "error": str(e)}))
            except Exception as e:  # unexpected bug: never crash the app
                self._observe(action, ok=False, error=f"internal error: {type(e).__name__}: {e}")
                self.emit(Event("action", f"`{action.tool}` crashed: {e}",
                                {"tool": action.tool, "args": action.args, "ok": False, "error": str(e)}))
        return True

    # ------------------------------------------------------------ helpers

    def _observe(self, action: Action, ok: bool, result: Any = None, error: str | None = None) -> None:
        obs: dict[str, Any] = {"tool": action.tool, "ok": ok}
        if ok:
            text = json.dumps(result, default=str)
            obs["result"] = result if len(text) <= MAX_OBS_CHARS else text[:MAX_OBS_CHARS] + "...(truncated)"
        else:
            obs["error"] = error
        self.observations.append(obs)
        self._log("action", tool=action.tool, args=action.args, ok=ok,
                  error=error, result_preview=json.dumps(result, default=str)[:300] if ok else None)

    def _log(self, event: str, **data: Any) -> None:
        log_event(self.session_id, event, **data)
