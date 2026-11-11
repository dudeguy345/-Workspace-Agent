"""The actions the agent can execute, grouped by domain.

Each tool has:
  * a pydantic model that validates its arguments,
  * a flag (or function) saying whether it is destructive (needs confirmation),
  * a plain Python implementation.

All file access is locked inside a per-session workspace folder, so the LLM
cannot touch anything else on the machine.
"""
from __future__ import annotations

import ast
import json
import operator
import platform
import shutil
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

MAX_FILE_BYTES = 100_000
HTTP_TIMEOUT = 10


class ToolError(Exception):
    """Raised for expected failures; the message is shown to the user and the LLM."""


@dataclass
class ToolContext:
    workspace: Path

    @property
    def db_path(self) -> Path:
        return self.workspace / ".tasks.db"


# ---------------------------------------------------------------- helpers

def safe_path(ctx: ToolContext, rel: str) -> Path:
    """Resolve `rel` inside the workspace or refuse."""
    root = ctx.workspace.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ToolError(f"Path '{rel}' is outside the workspace and was blocked.")
    if target.name == ".tasks.db":
        raise ToolError("The task database file cannot be accessed directly; use the task tools.")
    return target


def _db(ctx: ToolContext) -> sqlite3.Connection:
    conn = sqlite3.connect(ctx.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'medium',
            due TEXT,
            done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "workspace-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # network errors, HTTP errors, bad JSON
        raise ToolError(f"External API request failed: {e}") from e


# ---------------------------------------------------------------- file tools

class ListFilesArgs(BaseModel):
    path: str = "."


def list_files(ctx: ToolContext, a: ListFilesArgs) -> dict:
    p = safe_path(ctx, a.path)
    if not p.is_dir():
        raise ToolError(f"'{a.path}' is not a folder.")
    items = []
    for child in sorted(p.iterdir()):
        if child.name.startswith("."):
            continue
        items.append(
            {
                "name": child.name + ("/" if child.is_dir() else ""),
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return {"path": a.path, "items": items}


class ReadFileArgs(BaseModel):
    path: str


def read_file(ctx: ToolContext, a: ReadFileArgs) -> dict:
    p = safe_path(ctx, a.path)
    if not p.is_file():
        raise ToolError(f"File '{a.path}' does not exist.")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ToolError("File is too large to read (limit 100 KB).")
    return {"path": a.path, "content": p.read_text(encoding="utf-8", errors="replace")}


class WriteFileArgs(BaseModel):
    path: str
    content: str = Field(..., max_length=MAX_FILE_BYTES)
    overwrite: bool = False


def write_file(ctx: ToolContext, a: WriteFileArgs) -> dict:
    p = safe_path(ctx, a.path)
    if p.exists() and not a.overwrite:
        raise ToolError(f"'{a.path}' already exists. Set overwrite=true to replace it.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(a.content, encoding="utf-8")
    return {"path": a.path, "bytes_written": len(a.content.encode("utf-8"))}


class AppendFileArgs(BaseModel):
    path: str
    content: str = Field(..., max_length=MAX_FILE_BYTES)


def append_file(ctx: ToolContext, a: AppendFileArgs) -> dict:
    p = safe_path(ctx, a.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(a.content)
    return {"path": a.path, "bytes_appended": len(a.content.encode("utf-8"))}


class DeletePathArgs(BaseModel):
    path: str


def delete_path(ctx: ToolContext, a: DeletePathArgs) -> dict:
    p = safe_path(ctx, a.path)
    if p == ctx.workspace.resolve():
        raise ToolError("Refusing to delete the whole workspace.")
    if not p.exists():
        raise ToolError(f"'{a.path}' does not exist.")
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"deleted": a.path}


class MakeDirArgs(BaseModel):
    path: str


def make_dir(ctx: ToolContext, a: MakeDirArgs) -> dict:
    p = safe_path(ctx, a.path)
    p.mkdir(parents=True, exist_ok=True)
    return {"created": a.path}


# ---------------------------------------------------------------- database tools

class AddTaskArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    priority: Literal["low", "medium", "high"] = "medium"
    due: Optional[str] = Field(None, description="YYYY-MM-DD")


def add_task(ctx: ToolContext, a: AddTaskArgs) -> dict:
    if a.due:
        try:
            datetime.strptime(a.due, "%Y-%m-%d")
        except ValueError as e:
            raise ToolError("due must be in YYYY-MM-DD format") from e
    with _db(ctx) as conn:
        cur = conn.execute(
            "INSERT INTO tasks (title, priority, due, created_at) VALUES (?, ?, ?, ?)",
            (a.title, a.priority, a.due, datetime.now().isoformat(timespec="seconds")),
        )
    return {"added_id": cur.lastrowid, "title": a.title}


class ListTasksArgs(BaseModel):
    status: Literal["open", "done", "all"] = "open"


def list_tasks(ctx: ToolContext, a: ListTasksArgs) -> dict:
    sql = "SELECT id, title, priority, due, done FROM tasks"
    if a.status == "open":
        sql += " WHERE done = 0"
    elif a.status == "done":
        sql += " WHERE done = 1"
    sql += " ORDER BY done, CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, due"
    with _db(ctx) as conn:
        rows = [dict(r) for r in conn.execute(sql)]
    for r in rows:
        r["done"] = bool(r["done"])
    return {"status": a.status, "tasks": rows}


class TaskIdArgs(BaseModel):
    id: int


def complete_task(ctx: ToolContext, a: TaskIdArgs) -> dict:
    with _db(ctx) as conn:
        cur = conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (a.id,))
    if cur.rowcount == 0:
        raise ToolError(f"No task with id {a.id}.")
    return {"completed_id": a.id}


def delete_task(ctx: ToolContext, a: TaskIdArgs) -> dict:
    with _db(ctx) as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (a.id,))
    if cur.rowcount == 0:
        raise ToolError(f"No task with id {a.id}.")
    return {"deleted_id": a.id}


# ---------------------------------------------------------------- external API tools

class WeatherArgs(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)


def get_weather(ctx: ToolContext, a: WeatherArgs) -> dict:
    q = urllib.parse.urlencode({"name": a.city, "count": 1})
    geo = _http_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}")
    if not geo.get("results"):
        raise ToolError(f"Could not find a city named '{a.city}'.")
    place = geo["results"][0]
    q = urllib.parse.urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 3,
            "timezone": "auto",
        }
    )
    wx = _http_json(f"https://api.open-meteo.com/v1/forecast?{q}")
    return {
        "location": f"{place['name']}, {place.get('country', '')}",
        "current": wx.get("current"),
        "daily": wx.get("daily"),
        "units": wx.get("current_units"),
    }


class ExchangeArgs(BaseModel):
    base: str = Field(..., min_length=3, max_length=3)
    target: str = Field(..., min_length=3, max_length=3)
    amount: float = 1.0


def convert_currency(ctx: ToolContext, a: ExchangeArgs) -> dict:
    q = urllib.parse.urlencode(
        {"from": a.base.upper(), "to": a.target.upper(), "amount": a.amount}
    )
    data = _http_json(f"https://api.frankfurter.app/latest?{q}")
    if "rates" not in data:
        raise ToolError(f"Currency lookup failed: {data}")
    return {
        "amount": a.amount,
        "base": a.base.upper(),
        "result": data["rates"].get(a.target.upper()),
        "target": a.target.upper(),
        "date": data.get("date"),
    }


# ---------------------------------------------------------------- system tools

class NoArgs(BaseModel):
    pass


def system_info(ctx: ToolContext, a: NoArgs) -> dict:
    files = [p for p in ctx.workspace.rglob("*") if p.is_file() and p.name != ".tasks.db"]
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "workspace_files": len(files),
        "workspace_bytes": sum(p.stat().st_size for p in files),
    }


_OPS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("Exponent too large.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("Only numbers and + - * / // % ** ( ) are allowed.")


class CalcArgs(BaseModel):
    expression: str = Field(..., max_length=200)


def calculate(ctx: ToolContext, a: CalcArgs) -> dict:
    """Safe arithmetic — parses the expression instead of calling eval()."""
    try:
        tree = ast.parse(a.expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"Invalid expression: {e.msg}") from e
    try:
        return {"expression": a.expression, "result": _eval_node(tree.body)}
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e


# ---------------------------------------------------------------- registry

@dataclass
class Tool:
    name: str
    domain: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[ToolContext, Any], dict]
    destructive: bool | Callable[[ToolContext, BaseModel], bool] = False

    def is_destructive(self, ctx: ToolContext, args: BaseModel) -> bool:
        if callable(self.destructive):
            return self.destructive(ctx, args)
        return self.destructive


def _write_is_destructive(ctx: ToolContext, a: WriteFileArgs) -> bool:
    try:
        return a.overwrite and safe_path(ctx, a.path).exists()
    except ToolError:
        return False


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("list_files", "file", "List files in a workspace folder.", ListFilesArgs, list_files),
        Tool("read_file", "file", "Read a text file.", ReadFileArgs, read_file),
        Tool("write_file", "file", "Create a text file (overwrite=true replaces an existing file).",
             WriteFileArgs, write_file, _write_is_destructive),
        Tool("append_file", "file", "Append text to a file (creates it if missing).", AppendFileArgs, append_file),
        Tool("make_dir", "file", "Create a folder.", MakeDirArgs, make_dir),
        Tool("delete_path", "file", "Delete a file or folder.", DeletePathArgs, delete_path, True),
        Tool("add_task", "database", "Add a to-do task.", AddTaskArgs, add_task),
        Tool("list_tasks", "database", "List tasks (status: open | done | all).", ListTasksArgs, list_tasks),
        Tool("complete_task", "database", "Mark a task done by id.", TaskIdArgs, complete_task),
        Tool("delete_task", "database", "Permanently delete a task by id.", TaskIdArgs, delete_task, True),
        Tool("get_weather", "api", "Current weather + 3-day forecast for a city.", WeatherArgs, get_weather),
        Tool("convert_currency", "api", "Convert money using live exchange rates (ISO codes).",
             ExchangeArgs, convert_currency),
        Tool("system_info", "system", "OS, Python version, server time, workspace usage.", NoArgs, system_info),
        Tool("calculate", "system", "Evaluate an arithmetic expression exactly.", CalcArgs, calculate),
    ]
}


def tool_catalog() -> str:
    """Human/LLM-readable list of tools with their JSON argument schemas."""
    lines = []
    for t in TOOLS.values():
        props = t.args_model.model_json_schema().get("properties", {})
        required = set(t.args_model.model_json_schema().get("required", []))
        args = ", ".join(
            f"{k}{'' if k in required else '?'}: {v.get('type', v.get('enum', 'any'))}"
            for k, v in props.items()
        ) or "none"
        flag = " [DESTRUCTIVE - user must confirm]" if t.destructive is True else ""
        lines.append(f"- {t.name} ({t.domain}) args: {{{args}}} — {t.description}{flag}")
    return "\n".join(lines)
