"""Append-only audit log (JSON Lines). Every LLM call and action is recorded."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "audit.jsonl"
_lock = threading.Lock()


def log_event(session_id: str, event: str, **data: Any) -> dict:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session_id,
        "event": event,
        **data,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def read_events(session_id: str | None = None, limit: int = 200) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    out = []
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id is None or rec.get("session") == session_id:
                out.append(rec)
    return out[-limit:]
