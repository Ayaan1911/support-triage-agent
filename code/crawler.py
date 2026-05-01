"""
crawler.py — Support corpus crawler for the Multi-Domain Support Triage Agent.

Crawls support documentation from HackerRank, Claude, and Visa support sites
and saves clean text to corpus/{domain}/{slug}.txt.

Usage (via main.py):
    python main.py --crawl [--force]
"""

import os
import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMAINS = {
    "hackerrank": "https://support.hackerrank.com/",
    "claude": "https://support.claude.com/en/",
    "visa": "https://www.visa.co.in/support.html",
}

MAX_DEPTH = 2
REQUEST_DELAY = 1.0  # seconds between requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TriageAgentBot/1.0; +support-triage-research)"
    )
}
TIMEOUT = 15  # request timeout in seconds

# Tags whose text content we want to extract
CONTENT_TAGS = ["article", "main", "p", "h1", "h2", "h3", "h4"]

# Patterns to skip
SKIP_PATTERNS = re.compile(
    r"\.(pdf|zip|png|jpg|jpeg|gif|svg|mp4|webm|ico|css|js)$"
    r"|/login|/signin|/signup|/register|/auth|/oauth|/account/new",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(url: str) -> str:
    """Convert a URL to a safe filename slug."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    # Remove unsafe characters
    slug = re.sub(r"[^\w\-]", "_", path)
    slug = re.sub(r"_+", "_", slug)
    return slug[:200]  # cap length


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract clean text from configured content tags."""
    parts = []
    # Try structured content tags first
    for tag_name in CONTENT_TAGS:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)

    # Fallback: if nothing found, grab body text
    if not parts and soup.body:
        parts.append(soup.body.get_text(separator=" ", strip=True))

    # Deduplicate consecutive duplicate lines
    lines = []
    seen = set()
    for part in parts:
        for line in part.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                lines.append(stripped)

    return "\n".join(lines)


def _is_internal(base_url: str, link_url: str) -> bool:
    """Return True if link_url is on the same host as base_url."""
    base_host = urlparse(base_url).netloc
    link_host = urlparse(link_url).netloc
    # Allow empty netloc (relative links already resolved) or exact match
    return (not link_host) or (link_host == base_host)


def _should_skip(url: str) -> bool:
    """Return True if this URL should be skipped."""
    return bool(SKIP_PATTERNS.search(url))


def _get_links(soup: BeautifulSoup, current_url: str, base_url: str):
    """Extract all valid internal links from a page."""
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:") or href.startswith("#"):
            continue
        full_url = urljoin(current_url, href)
        # Strip fragment
        full_url = full_url.split("#")[0]
        if _is_internal(base_url, full_url) and not _should_skip(full_url):
            links.add(full_url)
    return links


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------


def crawl_domain(
    domain_name: str,
    start_url: str,
    corpus_dir: Path,
    session: requests.Session,
) -> int:
    """
    BFS crawl a single domain up to MAX_DEPTH levels deep.
    Returns the number of pages saved.
    """
    domain_dir = corpus_dir / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    # Queue entries: (url, depth)
    queue: list[tuple[str, int]] = [(start_url, 0)]
    saved = 0

    logger.info("[%s] Starting crawl from: %s", domain_name, start_url)

    while queue:
        url, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        if _should_skip(url):
            logger.debug("[%s] Skipping (pattern match): %s", domain_name, url)
            continue

        # Fetch page
        try:
            logger.info("[%s] Fetching (depth=%d): %s", domain_name, depth, url)
            response = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("[%s] Failed to fetch %s: %s", domain_name, url, exc)
            time.sleep(REQUEST_DELAY)
            continue

        # Only process HTML
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug("[%s] Skipping non-HTML content: %s", domain_name, url)
            time.sleep(REQUEST_DELAY)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        text = _extract_text(soup)

        if len(text.strip()) < 50:
            logger.debug("[%s] Skipping near-empty page: %s", domain_name, url)
        else:
            slug = _slugify(url)
            out_path = domain_dir / f"{slug}.txt"
            # Append index to avoid collisions
            counter = 1
            while out_path.exists():
                out_path = domain_dir / f"{slug}_{counter}.txt"
                counter += 1
            out_path.write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
            saved += 1
            logger.info("[%s] Saved: %s -> %s", domain_name, url, out_path.name)

        # Enqueue child links if within depth limit
        if depth < MAX_DEPTH:
            links = _get_links(soup, url, start_url)
            for link in links:
                if link not in visited:
                    queue.append((link, depth + 1))

        time.sleep(REQUEST_DELAY)

    logger.info("[%s] Crawl complete. Pages saved: %d", domain_name, saved)
    return saved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def corpus_has_files(corpus_dir: Path) -> bool:
    """Return True if any .txt files exist in the corpus directory."""
    for domain_dir in corpus_dir.iterdir():
        if domain_dir.is_dir():
            if any(domain_dir.glob("*.txt")):
                return True
    return False


def run_crawler(corpus_dir: Path, force: bool = False) -> None:
    """
    Main crawler entry point.

    Args:
        corpus_dir: Path to the corpus/ directory.
        force:      If True, crawl even if corpus already contains files.
    """
    if corpus_dir.exists() and corpus_has_files(corpus_dir):
        if not force:
            print(
                "[crawler] Corpus already contains files. "
                "Use --force to re-crawl and overwrite."
            )
            return
        else:
            print("[crawler] --force flag detected. Re-crawling all domains...")

    corpus_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    total_saved = 0

    for domain_name, start_url in DOMAINS.items():
        try:
            saved = crawl_domain(domain_name, start_url, corpus_dir, session)
            total_saved += saved
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Unexpected error during crawl: %s", domain_name, exc)

    print(f"[crawler] Done. Total pages saved: {total_saved}")
