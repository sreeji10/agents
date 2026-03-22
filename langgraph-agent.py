from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b")
BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("api_key") or os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set `api_key` or `API_KEY` in your environment/.env file."
    )


def _user_agent_header() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }


@tool
def get_current_time(tz: str = "UTC") -> str:
    """Get current time in a timezone offset format (examples: UTC, +05:30, -04:00)."""
    if tz.upper() == "UTC":
        dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    if not re.fullmatch(r"[+-]\d{2}:\d{2}", tz):
        return "Invalid timezone offset. Use 'UTC' or format like '+05:30'."

    sign = 1 if tz[0] == "+" else -1
    hours, minutes = map(int, tz[1:].split(":"))
    if hours > 23 or minutes > 59:
        return "Invalid timezone offset. Hour must be <= 23 and minutes <= 59."

    offset = timezone(sign * timedelta(hours=hours, minutes=minutes))
    dt = datetime.now(offset)
    return dt.strftime(f"%Y-%m-%d %H:%M:%S {tz}")


@tool
def fetch_url(url: str, max_chars: int = 5000) -> str:
    """Fetch plain text content from a URL. Useful for reading docs/articles directly."""
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"

    req = urllib.request.Request(url, headers=_user_agent_header())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", body)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "html" not in content_type and not text:
        text = body[:max_chars]

    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    return f"Source: {url}\nContent-Type: {content_type or 'unknown'}\n\n{text}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo HTML and return top result links/snippets."""
    max_results = max(1, min(max_results, 10))
    search_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    req = urllib.request.Request(search_url, headers=_user_agent_header())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search failed: {e}"

    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="(?P<href>[^"]+)">(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet" href="[^"]+">(?P<snippet>.*?)</a>',
        re.S,
    )
    results = []
    for match in pattern.finditer(html):
        href = re.sub(r"\s+", " ", match.group("href")).strip()
        title = re.sub(r"<[^>]+>", "", match.group("title")).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group("snippet")).strip()
        href = urllib.parse.unquote(href)

        parsed = urllib.parse.urlparse(href)
        if parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            direct = qs.get("uddg", [None])[0]
            if direct:
                href = direct

        results.append((title, href, snippet))
        if len(results) >= max_results:
            break

    if not results:
        return "No results found."

    lines = []
    for i, (title, href, snippet) in enumerate(results, start=1):
        lines.append(f"{i}. {title}\nURL: {href}\nSnippet: {snippet}")

    return "\n\n".join(lines)


def build_graph():
    model = ChatOpenAI(model=MODEL_NAME, base_url=BASE_URL, api_key=API_KEY, temperature=0)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    system_prompt = (
        "You are a helpful research assistant. "
        "Use available tools for factual and web-related requests. "
        "Cite URLs when you used web tools. "
        "Be clear, practical, and concise. "
        f"Current UTC time: {now_utc}"
    )
    return create_react_agent(
        model=model,
        tools=[get_current_time, fetch_url, web_search],
        prompt=system_prompt,
    )


def to_langchain_messages(chat_history: list[dict[str, str]]) -> list[Any]:
    msgs: list[Any] = []
    for m in chat_history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def main() -> None:
    st.set_page_config(page_title="Research Assistant (LangGraph)")
    st.title("Research Assistant")
    st.caption("LangGraph + Streamlit, with your existing tools")

    with st.sidebar:
        st.subheader("Model")
        st.code(MODEL_NAME)
        st.caption(f"Base URL: {BASE_URL}")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.last_tool_trace = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_tool_trace" not in st.session_state:
        st.session_state.last_tool_trace = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.last_tool_trace:
        with st.expander("Last tool calls"):
            for item in st.session_state.last_tool_trace:
                st.markdown(f"`{item['tool']}` args: `{item['args']}`")

    prompt = st.chat_input("Ask anything...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        graph = build_graph()
        tool_trace: list[dict[str, str]] = []
        final_text = ""

        inputs = {"messages": to_langchain_messages(st.session_state.messages)}
        for event in graph.stream(inputs, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue

            last = messages[-1]
            if isinstance(last, AIMessage):
                if last.tool_calls:
                    for tool_call in last.tool_calls:
                        tool_trace.append(
                            {
                                "tool": tool_call.get("name", "unknown"),
                                "args": str(tool_call.get("args", {})),
                            }
                        )
                if isinstance(last.content, str) and last.content.strip():
                    final_text = last.content
                    placeholder.markdown(final_text)

        if not final_text:
            final_text = "I could not generate a response."
            placeholder.markdown(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})
    st.session_state.last_tool_trace = tool_trace


if __name__ == "__main__":
    main()
