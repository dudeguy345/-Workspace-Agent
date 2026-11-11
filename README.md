# 🤖 Workspace Agent

**CS4680 Prompt Engineering — AI Agent Project**

An AI agent that turns natural-language requests into **real actions**. Claude plans
the steps as validated JSON, the agent executes them across four domains, feeds the
results back to the model, and repeats until the task is done.

| Domain | Tools |
|---|---|
| 📁 File operations (sandboxed) | `list_files`, `read_file`, `write_file`, `append_file`, `make_dir`, `delete_path`⚠️ |
| 🗃️ Database (SQLite) | `add_task`, `list_tasks`, `complete_task`, `delete_task`⚠️ |
| 🌐 External APIs | `get_weather` (Open-Meteo), `convert_currency` (Frankfurter) |
| 🖥️ System | `system_info`, `calculate` (safe math parser) |

⚠️ = destructive; the agent pauses for your approval before running these.

## Problem
LLM chatbots only *describe* what to do. People still copy answers into files, to-do
apps and calculators by hand. This agent connects the model's reasoning to real
operations, while keeping a person in control of anything risky and logging every step.

## How it works
See [docs/how_it_works.md](docs/how_it_works.md) (also shown in the app's “How it works” tab).

```
agent/
  llm.py      LLM integration: Claude API calls, error + rate-limit handling, JSON parsing, self-correction
  prompts.py  System prompt: role, output contract, tool catalog, rules, few-shot examples
  schema.py   Pydantic models for the LLM's output
  tools.py    Action executors for each domain + safety (path sandbox, arg validation)
  core.py     Agent loop (plan → validate → confirm → execute → observe), max 6 steps
  audit.py    JSON Lines audit log (logs/audit.jsonl)
app.py        Web UI (Streamlit)
cli.py        Terminal UI
tests/        Offline tests using a scripted fake LLM
```

## Run locally (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # or put it in .streamlit/secrets.toml
streamlit run app.py                      # web UI
python cli.py                             # terminal UI
pytest -q                                 # tests (no API key needed)
```

Optional settings: `AGENT_MODEL` (default `claude-opus-5-5`), `AGENT_EFFORT` (`low`/`medium`/`high`, default `medium`).
