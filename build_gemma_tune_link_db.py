#!/usr/bin/env python3
"""Build an organized link database from the Gemma fine-tuning documentation page."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

SEED_SOURCE_URLS = [
    "https://ai.google.dev/gemma/docs",
    "https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html",
    "https://unsloth.ai/docs/models/gemma-4",
    "https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune",
    "https://unsloth.ai/docs/get-started/fine-tuning-llms-guide",
    "https://docs.openvino.ai/2026/about-openvino/performance-benchmarks/generative-ai-performance.html",
    "https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html",
    "https://docs.openvino.ai/2026/model-server/ovms_docs_llm_reference.html",
    "https://docs.openvino.ai/2026/model-server/ovms_demos_common_export.html",
    "https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html",
    "https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gemma4",
    "https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gemma3",
    "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/gemma4/README.md",
    "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/gemma3/README.md",
    "https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.Adapter.html",
    "https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.AdapterConfig.html",
    "https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.LLMPipeline.html",
    "https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.VLMPipeline.html",
    "https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.draft_model.html",
    "https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_speculative_decoding.html",
    "https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_rag.html",
    "https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_agent.html",
    "https://docs.openvino.ai/2026/model-server/ovms_demo_long_context.html",
    "https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression/4-bit-weight-quantization.html",
    "https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html",
    "https://docs.openvino.ai/2026/openvino-workflow-generative/genai-model-preparation.html",
]
DISCOVER_SIDEBAR_FROM = "https://ai.google.dev/gemma/docs/"
OUT_DIR = Path("data")
DB_PATH = OUT_DIR / "gemma_docs_links.sqlite"
JSON_PATH = OUT_DIR / "gemma_docs_links.json"
CSV_PATH = OUT_DIR / "gemma_docs_links.csv"
MD_PATH = OUT_DIR / "gemma_docs_links.md"

CATEGORY_RULES = [
    ("fine-tuning", ["tune", "tuning", "finetune", "fine-tuning", "lora", "peft", "axolotl", "unsloth", "xtuner", "llama_factory"]),
    ("training", ["train", "training", "data", "dataset", "kaggle", "colab"]),
    ("optimization", ["optimiz", "gke", "gpu", "distributed", "vertex", "deploy", "serving", "performance"]),
    ("development", ["docs", "api", "reference", "github", "cookbook", "keras", "jax", "transformers", "huggingface", "gemma"]),
    ("safety-policy", ["policy", "terms", "privacy", "responsible", "prohibited"]),
    ("community", ["community", "forum", "discord", "twitter", "youtube", "linkedin", "github"]),
]

@dataclass
class LinkRow:
    id: str
    source_url: str
    source_title: str
    fetched_at: str
    ordinal: int
    text: str
    href_raw: str
    url: str
    domain: str
    path: str
    fragment: str
    scheme: str
    is_internal_ai_google: bool
    is_same_page_anchor: bool
    document_area: str
    section: str
    categories: str
    rel: str
    target: str
    aria_label: str
    title: str
    context: str


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def classify_area(a: Tag, main: Tag | None) -> str:
    if main and (a is main or a in main.descendants):
        return "main_content"
    ancestors = " ".join([norm_space(str(x.get("class", ""))) + " " + (x.name or "") for x in a.parents if isinstance(x, Tag)])
    if "devsite-nav" in ancestors or "nav" in ancestors:
        return "navigation"
    if "devsite-footer" in ancestors or "footer" in ancestors:
        return "footer"
    if "devsite-header" in ancestors or "header" in ancestors:
        return "header"
    return "page_chrome"


def nearest_section(a: Tag) -> str:
    # Prefer preceding heading inside article/main.
    for prev in a.find_all_previous(["h1", "h2", "h3", "h4"]):
        txt = norm_space(prev.get_text(" ", strip=True))
        if txt:
            return txt
    return ""


def categorize(text: str, url: str, section: str) -> list[str]:
    hay = f"{text} {url} {section}".lower().replace("-", "_")
    cats = []
    for cat, needles in CATEGORY_RULES:
        if any(n.lower().replace("-", "_") in hay for n in needles):
            cats.append(cat)
    return cats or ["reference"]


def context_for(a: Tag) -> str:
    parent = a.find_parent(["p", "li", "td", "th", "h1", "h2", "h3", "h4"])
    if parent:
        return norm_space(parent.get_text(" ", strip=True))[:500]
    return norm_space(a.get_text(" ", strip=True))[:500]


def discover_source_urls() -> list[str]:
    """Return seed URLs plus every Gemma docs page linked from the docs sidebar/nav."""
    urls: list[str] = []

    def add(url: str) -> None:
        normalized = url.split("#", 1)[0].rstrip("/")
        if normalized and normalized not in urls:
            urls.append(normalized)

    for url in SEED_SOURCE_URLS:
        add(url)

    resp = requests.get(DISCOVER_SIDEBAR_FROM, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(DISCOVER_SIDEBAR_FROM, a["href"]).split("#", 1)[0]
        parsed = urlparse(abs_url)
        if parsed.netloc == "ai.google.dev" and parsed.path.startswith("/gemma/docs"):
            add(abs_url)

    return urls


def build(source_urls: list[str]) -> list[LinkRow]:
    OUT_DIR.mkdir(exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[LinkRow] = []

    for source_url in source_urls:
        resp = requests.get(source_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        source_title = norm_space(soup.title.get_text(" ", strip=True) if soup.title else "")
        main = soup.find("devsite-content") or soup.find("main") or soup.find("article")

        for i, a in enumerate(soup.find_all("a", href=True), start=1):
            href = a.get("href", "")
            abs_url = urljoin(source_url, href)
            parsed = urlparse(abs_url)
            text = norm_space(a.get_text(" ", strip=True)) or norm_space(a.get("aria-label", "")) or norm_space(a.get("title", ""))
            section = nearest_section(a)
            area = classify_area(a, main)
            cats = categorize(text, abs_url, section)
            stable = hashlib.sha1(f"{source_url}|{i}|{href}|{text}".encode()).hexdigest()[:12]
            rows.append(LinkRow(
                id=stable,
                source_url=source_url,
                source_title=source_title,
                fetched_at=fetched_at,
                ordinal=i,
                text=text,
                href_raw=href,
                url=abs_url,
                domain=parsed.netloc,
                path=parsed.path,
                fragment=parsed.fragment,
                scheme=parsed.scheme,
                is_internal_ai_google=parsed.netloc == "ai.google.dev" or (not parsed.netloc and href.startswith("/")),
                is_same_page_anchor=(parsed._replace(fragment="").geturl() == source_url and bool(parsed.fragment)) or href.startswith("#"),
                document_area=area,
                section=section,
                categories=", ".join(cats),
                rel=" ".join(a.get("rel", [])) if isinstance(a.get("rel"), list) else norm_space(a.get("rel", "")),
                target=norm_space(a.get("target", "")),
                aria_label=norm_space(a.get("aria-label", "")),
                title=norm_space(a.get("title", "")),
                context=context_for(a),
            ))
    return rows


def write_outputs(rows: list[LinkRow], source_urls: list[str]) -> None:
    data = [asdict(r) for r in rows]
    JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    fields = list(data[0].keys()) if data else []
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cols = ", ".join([f"{field} TEXT" for field in fields])
    con.execute(f"CREATE TABLE links ({cols})")
    con.execute("""
        CREATE TABLE source_pages (
            source_url TEXT PRIMARY KEY,
            source_title TEXT,
            fetched_at TEXT,
            link_instances INTEGER,
            main_content_links INTEGER,
            unique_outbound_urls INTEGER
        )
    """)
    con.executemany(
        f"INSERT INTO links ({', '.join(fields)}) VALUES ({', '.join(['?'] * len(fields))})",
        [[str(row.get(field, "")) for field in fields] for row in data],
    )
    source_rows = []
    for source_url in source_urls:
        page_rows = [r for r in rows if r.source_url == source_url]
        source_rows.append((
            source_url,
            page_rows[0].source_title if page_rows else "",
            page_rows[0].fetched_at if page_rows else "",
            len(page_rows),
            sum(1 for r in page_rows if r.document_area == "main_content"),
            len({r.url for r in page_rows}),
        ))
    con.executemany(
        "INSERT INTO source_pages VALUES (?, ?, ?, ?, ?, ?)",
        source_rows,
    )

    con.execute("CREATE INDEX idx_links_area ON links(document_area)")
    con.execute("CREATE INDEX idx_links_domain ON links(domain)")
    con.execute("CREATE INDEX idx_links_categories ON links(categories)")
    con.execute("CREATE INDEX idx_links_section ON links(section)")
    con.execute("""
        CREATE VIEW unique_links AS
        SELECT
            min(id) AS id,
            url,
            group_concat(DISTINCT source_url) AS source_urls,
            group_concat(DISTINCT text) AS labels,
            group_concat(DISTINCT domain) AS domains,
            group_concat(DISTINCT categories) AS categories,
            group_concat(DISTINCT document_area) AS document_areas,
            group_concat(DISTINCT section) AS sections,
            min(context) AS sample_context,
            count(*) AS occurrences
        FROM links
        GROUP BY url
    """)
    con.execute("""
        CREATE VIEW gemma_workbench_links AS
        SELECT * FROM links
        WHERE document_area = 'main_content'
           OR categories LIKE '%fine-tuning%'
           OR categories LIKE '%training%'
           OR categories LIKE '%optimization%'
           OR categories LIKE '%development%'
    """)
    con.commit()
    con.close()

    # Human-readable grouped view, focused on tuning/training/optimization/development.
    by_area: dict[str, list[LinkRow]] = {}
    for r in rows:
        by_area.setdefault(r.document_area, []).append(r)
    lines = [
        "# Gemma docs link database",
        "",
        f"Sources crawled: {len(source_urls)}",
        f"Fetched: {rows[0].fetched_at if rows else ''}",
        f"Total link instances: {len(rows)}",
        "",
        "Source pages:",
        *[f"- {url}" for url in source_urls],
        "",
        "Generated artifacts:",
        f"- SQLite: `{DB_PATH}`",
        f"- JSON: `{JSON_PATH}`",
        f"- CSV: `{CSV_PATH}`",
        "",
    ]
    for area in ["main_content", "navigation", "header", "footer", "page_chrome"]:
        area_rows = by_area.get(area, [])
        if not area_rows:
            continue
        lines += [f"## {area.replace('_', ' ').title()} ({len(area_rows)})", ""]
        current_section = None
        for r in area_rows:
            if r.section != current_section:
                current_section = r.section
                if current_section:
                    lines += [f"### {current_section}", ""]
            label = r.text or r.aria_label or r.title or "(no text)"
            lines.append(f"- **{label}** — {r.url}")
            lines.append(f"  - categories: {r.categories}; domain: {r.domain or '(relative)'}")
            if r.context and r.context != label:
                lines.append(f"  - context: {r.context}")
        lines.append("")
    MD_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    source_urls = discover_source_urls()
    rows = build(source_urls)
    write_outputs(rows, source_urls)
    print(f"crawled {len(source_urls)} source pages")
    print(f"wrote {len(rows)} link instances")
    print(DB_PATH)
    print(JSON_PATH)
    print(CSV_PATH)
    print(MD_PATH)
