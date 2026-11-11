"""Terminal UI. Run with:  python cli.py"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.core import Agent, Event

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspaces" / "cli"

C = {"dim": "\033[2m", "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
     "cyan": "\033[36m", "bold": "\033[1m", "end": "\033[0m"}


def show(e: Event) -> None:
    if e.kind == "thought":
        print(f"{C['dim']}  thinking: {e.text}{C['end']}")
    elif e.kind == "action":
        ok = e.data.get("ok")
        color = C["green"] if ok else C["red"]
        print(f"{color}  {'✔' if ok else '✘'} {e.text}{C['end']} {C['dim']}{json.dumps(e.data.get('args'))}{C['end']}")
    elif e.kind == "final":
        print(f"\n{C['cyan']}{e.text}{C['end']}\n")
    elif e.kind == "error":
        print(f"{C['red']}  error: {e.text}{C['end']}")


def main() -> None:
    if not WORKSPACE.exists() and (ROOT / "sample_workspace").exists():
        shutil.copytree(ROOT / "sample_workspace", WORKSPACE)
    agent = Agent(WORKSPACE, session_id="cli")
    agent.emit = show
    print(f"{C['bold']}Workspace Agent{C['end']} — workspace: {WORKSPACE}")
    print("Type a request in plain English. 'quit' to exit.\n")
    while True:
        try:
            text = input(f"{C['bold']}you> {C['end']}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        agent.submit(text)
        while agent.awaiting:
            a = agent.awaiting
            print(f"{C['yellow']}  ⚠ Destructive action: {a.tool} {json.dumps(a.args)}")
            if a.reason:
                print(f"    reason: {a.reason}")
            ans = input(f"    Approve? [y/N] {C['end']}").strip().lower()
            agent.resolve(ans in {"y", "yes"})


if __name__ == "__main__":
    main()
