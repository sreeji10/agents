import os
import re
import urllib
import urllib.parse
import urllib.request
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b")
API_KEY = os.getenv("api_key") or os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set `api_key` or `API_KEY` in your environment/.env file."
    )

llm = ChatNVIDIA(model=MODEL_NAME, nvidia_api_key=API_KEY)


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo HTML and return top result links/snippets.
    """
    max_results = max(1, min(max_results, 10))
    search_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

    req = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Search failed: {e}"

    # Parse top results from DuckDuckGo's HTML fallback page.
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

        # Convert DuckDuckGo redirect URL to direct URL when possible.
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


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> str:
    """
    Compatibility wrapper around DuckDuckGo search.
    """
    _ = topic, include_raw_content
    return web_search(query=query, max_results=max_results)


research_instructions = """
You are a helpful research assistant.
Use available tools for factual and web-related requests.
Cite URLs when you used web tools.
Be clear, practical, and concise.
"""

agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt=research_instructions,
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "When is kerala election?"}]}
    )
    print(result["messages"][-1].content)
