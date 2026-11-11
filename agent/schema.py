"""Data models for what the LLM is allowed to say.

Everything the LLM returns is parsed into these models before anything runs.
If the output doesn't fit, the agent never executes it.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    """One concrete operation the LLM wants the agent to run."""

    tool: str = Field(..., description="Name of a registered tool")
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field("", description="Why this step is needed (shown to the user)")


class AgentResponse(BaseModel):
    """The single JSON object the LLM must return on every turn."""

    thought: str = Field(..., description="Short reasoning / plan for this turn")
    actions: list[Action] = Field(default_factory=list)
    final_answer: Optional[str] = Field(
        None, description="Set only when the task is complete; ends the loop"
    )

    @field_validator("actions")
    @classmethod
    def limit_actions(cls, v: list[Action]) -> list[Action]:
        if len(v) > 5:
            raise ValueError("at most 5 actions per turn")
        return v

    @property
    def done(self) -> bool:
        return self.final_answer is not None and not self.actions
