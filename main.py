"""
agent.py — Minimal function-calling agent for LM Studio / OpenAI-compatible APIs

Usage:
  1) python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
  2) pip install requests python-dotenv
  3) Create .env (see below)
  4) python agent.py "Add 41 and 1 using the add tool, then tell me the current time using get_time."

.env example:
  OPENAI_API_KEY=lm-studio                 # arbitrary for LM Studio
  OPENAI_BASE_URL=http://localhost:1234/v1 # LM Studio → Developer → Local Server
  MODEL=Your-Loaded-Model-Name            # e.g. Qwen2.5-7B-Instruct-GGUF

LM Studio: start local server (Developer → Local Server) to expose /v1/chat/completions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# -----------------------------
# Types
# -----------------------------
@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[Any] = None

@dataclass
class ToolSpec:
    type: str
    function: Dict[str, Any]

@dataclass
class ToolCall:
    id: str
    type: str
    function: Dict[str, Any]

# -----------------------------
# Config
# -----------------------------
load_dotenv()
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL = os.getenv("MODEL", "local-model")

SYSTEM_PROMPT = (
    "You are a practical software agent.\n"
    "- When a tool is relevant, call it once with minimal arguments.\n"
    "- If a tool call returns an error, explain the error and propose a fix.\n"
    "- Keep answers short and actionable."
)

# -----------------------------
# Tool registry (add your own functions here)
# -----------------------------

def tool_add(args: Dict[str, Any]) -> Dict[str, Any]:
    a = args.get("a")
    b = args.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("add: a and b must be numbers")
    return {"result": a + b}


def tool_get_time(_args: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime, timezone

    return {"iso": datetime.now(timezone.utc).isoformat()}


def tool_grep_text(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern")
    text = args.get("text", "")
    if not isinstance(pattern, str) or not isinstance(text, str):
        raise ValueError("grep_text: 'pattern' and 'text' must be strings")
    import re

    rx = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    matches = [
        {"match": m.group(0), "start": m.start(), "end": m.end()} for m in rx.finditer(text)
    ]
    return {"count": len(matches), "matches": matches}


TOOL_REGISTRY = {
    "add": tool_add,
    "get_time": tool_get_time,
    "grep_text": tool_grep_text,
}

TOOLS: List[ToolSpec] = [
    ToolSpec(
        type="function",
        function={
            "name": "add",
            "description": "Add two numbers and return their sum",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    ),
    ToolSpec(
        type="function",
        function={
            "name": "get_time",
            "description": "Get current time in ISO 8601 (UTC)",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ),
    ToolSpec(
        type="function",
        function={
            "name": "grep_text",
            "description": "Search for a regex pattern in the provided text and return matches",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "text": {"type": "string"}},
                "required": ["pattern", "text"],
                "additionalProperties": False,
            },
        },
    ),
]

# -----------------------------
# OpenAI-compatible /v1/chat/completions
# -----------------------------

def chat(messages: List[ChatMessage], tools: Optional[List[ToolSpec]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": [m.__dict__ for m in messages],
        "temperature": 0.2,
        "tool_choice": "auto",
    }
    if tools:
        payload["tools"] = [t.__dict__ for t in tools]

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    return choice.get("message") or {}


# -----------------------------
# Agent loop: one round of tool use + final answer
# -----------------------------

def run_agent(user_prompt: str) -> str:
    messages: List[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]

    first = chat(messages, TOOLS)
    if first:
        messages.append(ChatMessage(**first))

    tool_calls = first.get("tool_calls") if isinstance(first, dict) else None
    if tool_calls:
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", "{}")
            try:
                parsed = json.loads(raw_args)
            except Exception as e:
                parsed = {}
                result: Dict[str, Any] = {"error": f"Bad JSON args: {e}"}
            else:
                impl = TOOL_REGISTRY.get(name)
                if impl is None:
                    result = {"error": f"Tool not implemented: {name}"}
                else:
                    try:
                        result = impl(parsed)
                    except Exception as e:  # capture tool error and send back to model
                        result = {"error": str(e)}

            messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=tc.get("id"),
                    content=json.dumps(result, ensure_ascii=False),
                )
            )

        final = chat(messages)
        if final:
            messages.append(ChatMessage(**final))

    # Return last assistant content
    for m in reversed(messages):
        if m.role == "assistant" and m.content:
            return m.content
    return "[no content]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal function-calling agent")
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="User prompt")
    args = parser.parse_args()

    user_prompt = " ".join(args.prompt).strip()
    if not user_prompt:
        user_prompt = (
            "Please add 41 and 1 using the add tool, then tell me the current time using get_time."
        )

    try:
        out = run_agent(user_prompt)
    except Exception as e:
        print("Agent error:", e, file=sys.stderr)
        sys.exit(1)
    print("\n--- AGENT OUTPUT ---\n" + out)


if __name__ == "__main__":
    main()
