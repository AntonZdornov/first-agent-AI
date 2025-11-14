"""
agent_langchain.py — LangChain agent that uses your local LM Studio (OpenAI-compatible) and calls Python tools.

Setup:
  pip install -U langchain langchain-openai duckduckgo-search requests beautifulsoup4 python-dotenv

.env (example):
  OPENAI_BASE_URL=http://localhost:1234/v1
  OPENAI_API_KEY=lm-studio
  MODEL=Your-Loaded-Model-Name

Run:
  python agent_langchain.py "Find 2 best recent articles about asyncio in Python and summarize them in 3 bullets."
"""
from __future__ import annotations

import os
import argparse
from typing import List

from dotenv import load_dotenv

# LangChain core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
# Memory imports
# from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Optional web tools deps
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

load_dotenv()

# -----------------------------
# Tools (via @tool decorator)
# -----------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def get_time() -> str:
    """Return current UTC time in ISO 8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


@tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo and return a list of {title, url, snippet}.

    Args:
        query: search query.
        max_results: 1..10, how many results to return.
    """
    max_results = max(1, min(int(max_results), 10))
    out: List[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
            })
    return out


@tool
def web_get(url: str, max_chars: int = 4000, timeout: int = 10) -> dict:
    """Fetch a URL and return a plain‑text preview of the page content.

    Args:
        url: http/https link to fetch.
        max_chars: truncate the extracted text to this many characters.
        timeout: request timeout in seconds.
    """
    if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
        raise ValueError("web_get: valid http(s) URL required")
    resp = requests.get(url, timeout=int(timeout), headers={"User-Agent": "LangChain-Agent/0.1"})
    resp.raise_for_status()
    text = _extract_text(resp.text)
    preview = text[: int(max_chars)]
    return {
        "url": url,
        "chars": len(text),
        "preview": preview,
        "truncated": len(text) > int(max_chars),
    }


TOOLS = [add, get_time, web_search, web_get]

# -----------------------------
# Model (LM Studio via OpenAI‑compatible base_url)
# -----------------------------

llm = ChatOpenAI(
    model=os.getenv("MODEL", "local"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    temperature=0.2,
)

# -----------------------------
# Prompt + Agent
# -----------------------------

SYSTEM = (
    "You are a pragmatic software agent. Use tools when helpful. "
    "Cite URLs you used from web_get. Keep answers concise."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # ← обязательный
    ("user", "{input}"),
])

agent = create_tool_calling_agent(llm, TOOLS, prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# --- Simple in-memory chat history (per session_id) ---
# _store: dict[str, InMemoryChatMessageHistory] = {}

# def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in _store:
#         _store[session_id] = InMemoryChatMessageHistory()
#     return _store[session_id]

# запись в файл
def _get_session_history(session_id: str):
    return FileChatMessageHistory(f"./memory_{session_id}.json")

executor_with_history = RunnableWithMessageHistory(
    executor,
    _get_session_history,
    input_messages_key="input",        # matches prompt variable
    history_messages_key="chat_history" # matches MessagesPlaceholder
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs=argparse.REMAINDER, help="User prompt")
    args = parser.parse_args()
    user_prompt = " ".join(args.prompt).strip() or "Add 41 and 1 using the add tool, then tell me the current time."

    # Minimal run (stateless). You can also pass chat_history=[...] for memory.
    # Use a stable session_id to keep memory across calls
    session_id = os.getenv("SESSION_ID", "anton")
    print(f'session_id={session_id}')
    result = executor_with_history.invoke(
        {"input": user_prompt},
        config={"configurable": {"session_id": session_id}},
    )
    print("\n--- AGENT OUTPUT ---\n" + str(result.get("output")))


if __name__ == "__main__":
    main()


# export SESSION_ID=anton
# python agent_langchain.py "Привет, запомни что я люблю TypeScript."
# python agent_langchain.py "Что я говорил о любимом языке?"
# echo $SESSION_ID