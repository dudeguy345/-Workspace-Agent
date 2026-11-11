### The problem
Chatbots *tell* you what to do; you still have to do it. Workspace Agent closes that gap:
you describe a task in plain English and it **plans, executes, and reports** real actions —
safely, with an audit trail.

### Architecture
```
 you ──► UI (Streamlit / terminal)
           │
           ▼
      Agent loop (agent/core.py) ◄──────────────┐
           │ history + observations              │ observations
           ▼                                     │
      LLM module (agent/llm.py) ── Claude API    │
           │ raw text                            │
           ▼                                     │
      Parser + validator (JSON → pydantic)       │
           │ valid plan        invalid → retry with error (self-correction)
           ▼                                     │
      Safety gate: destructive? → ask user       │
           │                                     │
           ▼                                     │
      Executor (agent/tools.py) ─────────────────┘
        📁 files  🗃️ SQLite tasks  🌐 weather / currency APIs  🖥️ system / calculator
           │
           ▼
      Audit log (logs/audit.jsonl)
```

### Prompt engineering techniques
| Technique | Where |
|---|---|
| Role prompting & scope | system prompt intro |
| Strict JSON output contract | "Output contract" section |
| Tool grounding (catalog generated from code) | `tool_catalog()` |
| Plan-before-act (`thought` field) | schema + examples |
| Few-shot examples (single-step, multi-step, refusal) | "Examples" section |
| ReAct loop (observe → re-plan) | `agent/core.py` |
| Self-correction on invalid output | `correction_prompt()` |
| Safety rules & out-of-scope handling | "Rules" section |

### Safety
- Every LLM reply is validated (schema, known tool, typed arguments) **before** anything runs.
- Files are sandboxed to a per-session workspace (`..` and absolute paths are blocked).
- Deletes and overwrites require your explicit approval.
- Calculator parses math instead of using `eval`; no arbitrary shell commands.
- Max 6 LLM steps per request, local rate limit, SDK retries with backoff.
- Everything is logged to `logs/audit.jsonl`.
