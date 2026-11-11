"""Web UI (Streamlit). Run locally with:  streamlit run app.py"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import streamlit as st

from agent.audit import read_events
from agent.core import Agent, Event
from agent.tools import TOOLS

ROOT = Path(__file__).resolve().parent
WORKSPACES = ROOT / "workspaces"
SAMPLE = ROOT / "sample_workspace"

EXAMPLES = [
    "What files do I have? Summarize meeting_notes.md",
    "Turn the action items in meeting_notes.md into tasks",
    "What's the weather in Pomona, CA? Save it to notes/weather.md",
    "Convert 250 USD to EUR and JPY",
    "Show my open tasks, then mark the highest-priority one done",
    "Delete budget.csv",
]

DOMAIN_ICON = {"file": "📁", "database": "🗃️", "api": "🌐", "system": "🖥️"}

st.set_page_config(page_title="Workspace Agent", page_icon="🤖", layout="wide")


# ---------------------------------------------------------------- session setup

def get_api_key() -> str | None:
    if st.session_state.get("user_key"):
        return st.session_state.user_key
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.getenv("ANTHROPIC_API_KEY")


def new_session() -> None:
    sid = uuid.uuid4().hex[:8]
    ws = WORKSPACES / sid
    if SAMPLE.exists():
        shutil.copytree(SAMPLE, ws, dirs_exist_ok=True)
    st.session_state.sid = sid
    st.session_state.workspace = ws
    st.session_state.transcript = []
    st.session_state.agent = None


if "sid" not in st.session_state:
    new_session()


def get_agent() -> Agent | None:
    if st.session_state.agent is None:
        key = get_api_key()
        if not key:
            return None
        agent = Agent(st.session_state.workspace, api_key=key, session_id=st.session_state.sid)
        st.session_state.agent = agent
    agent = st.session_state.agent
    agent.emit = lambda e: st.session_state.transcript.append(e)  # rebind each rerun
    return agent


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🤖 Workspace Agent")
    st.caption("Natural language → real actions on files, a task database, web APIs and system utilities.")

    if not get_api_key():
        st.warning("No API key configured.")
    key = st.text_input("Anthropic API key (optional override)", type="password",
                        help="Used only in your browser session; never written to disk.")
    if key and key != st.session_state.get("user_key"):
        st.session_state.user_key = key
        st.session_state.agent = None

    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.queued = ex

    st.subheader("📂 Your workspace")
    ws: Path = st.session_state.workspace
    files = sorted(p for p in ws.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        st.caption("(empty)")
    for p in files:
        rel = p.relative_to(ws).as_posix()
        with st.expander(f"{rel}  ·  {p.stat().st_size} B"):
            st.code(p.read_text(encoding="utf-8", errors="replace")[:3000], language=None)

    if st.button("🔄 Reset session", use_container_width=True):
        new_session()
        st.rerun()

    with st.expander("Available tools"):
        for t in TOOLS.values():
            danger = " ⚠️" if t.destructive is True else ""
            st.markdown(f"{DOMAIN_ICON[t.domain]} `{t.name}`{danger} — {t.description}")


# ---------------------------------------------------------------- rendering

def render_event(e: Event) -> None:
    if e.kind == "thought":
        st.caption(f"💭 {e.text}")
    elif e.kind == "action":
        d = e.data
        icon = "⛔" if d.get("denied") else ("✅" if d.get("ok") else "❌")
        domain = DOMAIN_ICON.get(TOOLS[d["tool"]].domain, "") if d.get("tool") in TOOLS else ""
        with st.expander(f"{icon} {domain} {e.text}", expanded=not d.get("ok")):
            if d.get("reason"):
                st.markdown(f"*Why:* {d['reason']}")
            st.markdown("**Arguments**")
            st.json(d.get("args", {}))
            if "result" in d:
                st.markdown("**Result**")
                st.json(d["result"])
            if d.get("error"):
                st.error(d["error"])
    elif e.kind == "final":
        st.markdown(e.text)
    elif e.kind == "error":
        st.error(f"⚠️ {e.text}")
    elif e.kind == "confirm":
        st.warning(f"🛑 {e.text}")


def render_transcript() -> None:
    group: list[Event] = []

    def flush():
        if group:
            with st.chat_message("assistant"):
                for ev in group:
                    render_event(ev)
            group.clear()

    for e in st.session_state.transcript:
        if e.kind == "user":
            flush()
            with st.chat_message("user"):
                st.markdown(e.text)
        else:
            group.append(e)
    flush()


# ---------------------------------------------------------------- main

chat_tab, log_tab, about_tab = st.tabs(["💬 Agent", "📜 Audit log", "ℹ️ How it works"])

with chat_tab:
    agent = get_agent()
    if agent is None:
        st.info("Add an Anthropic API key in the sidebar (or in `.streamlit/secrets.toml`) to start.")

    if not st.session_state.transcript:
        st.markdown("#### What can I do for you?")
        st.caption("Ask in plain English. I plan the steps, run them, and show you everything I did. "
                   "Anything destructive waits for your approval.")

    render_transcript()

    # Pending confirmation card
    if agent and agent.awaiting:
        a = agent.awaiting
        with st.container(border=True):
            st.markdown(f"### 🛑 Confirm: `{a.tool}`")
            if a.reason:
                st.markdown(f"**Reason given by the agent:** {a.reason}")
            st.json(a.args)
            c1, c2 = st.columns(2)
            if c1.button("✅ Approve", type="primary", use_container_width=True):
                with st.spinner("Running..."):
                    agent.resolve(True)
                st.rerun()
            if c2.button("❌ Deny", use_container_width=True):
                with st.spinner("Continuing..."):
                    agent.resolve(False)
                st.rerun()

    prompt = st.chat_input("e.g. Create a packing list for a 3-day trip to Seattle",
                           disabled=agent is None or agent.busy)
    prompt = prompt or st.session_state.pop("queued", None)
    if prompt and agent and not agent.busy:
        st.session_state.transcript.append(Event("user", prompt))
        with st.spinner("Thinking and acting..."):
            agent.submit(prompt)
        st.rerun()

with log_tab:
    st.caption("Every request, LLM plan, confirmation and action is appended to `logs/audit.jsonl`.")
    events = read_events(st.session_state.sid)
    if not events:
        st.write("No events yet.")
    else:
        st.dataframe(
            [{"time": e["ts"], "event": e["event"],
              "details": json.dumps({k: v for k, v in e.items() if k not in ("ts", "session", "event")},
                                    default=str)[:300]}
             for e in reversed(events)],
            use_container_width=True, hide_index=True,
        )
        st.download_button("Download log (JSONL)",
                           "\n".join(json.dumps(e, default=str) for e in events),
                           file_name=f"audit_{st.session_state.sid}.jsonl")

with about_tab:
    st.markdown((ROOT / "docs" / "how_it_works.md").read_text(encoding="utf-8"))
