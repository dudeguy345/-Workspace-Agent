"""LLM integration: talks to Claude, handles API errors / rate limits,
and turns raw text into a validated AgentResponse (with self-correction)."""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque

import anthropic
from pydantic import ValidationError

from .prompts import build_system_prompt, correction_prompt
from .schema import AgentResponse
from .tools import TOOLS

MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("AGENT_EFFORT", "medium")  # low | medium | high
MAX_CORRECTIONS = 2


class LLMError(Exception):
    """A user-facing error from the LLM layer."""


class RateLimiter:
    """Simple sliding-window limiter so one user can't burn the API key."""

    def __init__(self, max_calls: int = 15, per_seconds: float = 60.0):
        self.max_calls, self.per = max_calls, per_seconds
        self.calls: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.per:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            wait = int(self.per - (now - self.calls[0])) + 1
            raise LLMError(f"Local rate limit reached. Please wait about {wait}s and try again.")
        self.calls.append(now)


# ---------------------------------------------------------------- parsing / validation

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's text, tolerating fences or stray prose."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e.msg} at position {e.pos})") from e
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON must be an object")
    return obj


def parse_response(text: str) -> AgentResponse:
    """Parse + validate. Raises ValueError with a message the model can act on."""
    obj = extract_json(text)
    try:
        resp = AgentResponse.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"JSON does not match the schema: {e.errors(include_url=False)}") from e

    if not resp.actions and resp.final_answer is None:
        raise ValueError("either provide actions or a final_answer")
    for i, act in enumerate(resp.actions):
        tool = TOOLS.get(act.tool)
        if tool is None:
            raise ValueError(f"actions[{i}] uses unknown tool '{act.tool}'. Valid: {sorted(TOOLS)}")
        try:
            tool.args_model.model_validate(act.args)
        except ValidationError as e:
            raise ValueError(
                f"actions[{i}] ({act.tool}) has invalid args: {e.errors(include_url=False)}"
            ) from e
    return resp


# ---------------------------------------------------------------- client

class LLMClient:
    def __init__(self, api_key: str | None = None, on_event=None):
        # max_retries: the SDK retries 429/5xx/connection errors with exponential backoff.
        self.client = anthropic.Anthropic(api_key=api_key or None, max_retries=3, timeout=120.0)
        self.system = build_system_prompt()
        self.limiter = RateLimiter()
        self.on_event = on_event or (lambda *a, **k: None)

    def _call(self, messages: list[dict]) -> str:
        self.limiter.check()
        try:
            resp = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=self.system,
                messages=messages,
                output_config={"effort": EFFORT},
                cache_control={"type": "ephemeral"},
                # If a safety classifier declines, retry server-side on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Invalid or missing Anthropic API key.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError("This API key is not allowed to use the model.") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Model '{MODEL}' was not found.") from e
        except anthropic.RateLimitError as e:
            retry = e.response.headers.get("retry-after", "a few")
            raise LLMError(f"Claude API rate limit hit. Try again in {retry} seconds.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Bad request to the API: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}. Please retry shortly.") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Claude API. Check your internet connection.") from e

        self.on_event(
            "llm_call",
            model=resp.model,
            stop_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            request_id=getattr(resp, "_request_id", None),
        )
        if resp.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if resp.stop_reason == "max_tokens":
            raise LLMError("The model's reply was cut off (too long). Try a smaller request.")
        return "".join(b.text for b in resp.content if b.type == "text")

    def next_step(self, history: list[dict]) -> tuple[AgentResponse, str]:
        """Ask for the next step. Retries with error feedback if the output is invalid.

        Returns the validated response and the raw text to store in history.
        """
        attempt_msgs = list(history)
        last_error = ""
        for attempt in range(MAX_CORRECTIONS + 1):
            text = self._call(attempt_msgs)
            try:
                return parse_response(text), text
            except ValueError as e:
                last_error = str(e)
                self.on_event("invalid_llm_output", attempt=attempt, error=last_error, raw=text[:2000])
                attempt_msgs = attempt_msgs + [
                    {"role": "assistant", "content": text or "(empty)"},
                    {"role": "user", "content": correction_prompt(last_error)},
                ]
        raise LLMError(f"The model did not return a valid plan after {MAX_CORRECTIONS + 1} tries: {last_error}")
