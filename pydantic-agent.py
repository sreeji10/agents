import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b")
BASE_URL = os.getenv("BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("api_key") or os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set `api_key` or `API_KEY` in your environment/.env file."
    )

model = OpenAIChatModel(
    model_name=MODEL_NAME,
    provider=OpenAIProvider(base_url=BASE_URL, api_key=API_KEY),
)

agent = Agent(
    model=model,
    instructions=(
        "You are a helpful research assistant. "
        "Use available tools for factual and web-related requests. "
        "Cite URLs when you used web tools. "
        "Be clear, practical, and concise."
    ),
)


@agent.instructions
def runtime_context() -> str:
    """Provide runtime context to the model each run."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"Current UTC time: {now_utc}"


@agent.tool_plain
def get_current_time(tz: str = "UTC") -> str:
    """
    Get current time in a timezone offset format (examples: UTC, +05:30, -04:00).
    """
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


@agent.tool_plain
def fetch_url(url: str, max_chars: int = 5000) -> str:
    """
    Fetch plain text content from a URL. Useful for reading docs/articles directly.
    """
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"

    req = urllib.request.Request(
        url,
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
            content_type = (resp.headers.get("Content-Type") or "").lower()
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    # Minimal HTML to text cleanup, enough for quick model consumption.
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", body)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if "html" not in content_type and not text:
        text = body[:max_chars]

    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    return f"Source: {url}\nContent-Type: {content_type or 'unknown'}\n\n{text}"


@agent.tool_plain
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


# Keep web UI behavior using the already-configured model object.
app = agent.to_web()
