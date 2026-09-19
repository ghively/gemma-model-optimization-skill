#!/usr/bin/env python3
"""Query the Gemma docs link database bundled with this skill."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "gemma_docs_links.sqlite"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("terms", nargs="*", help="Search terms. All terms must match url/label/context/categories/source.")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--source", help="Restrict to a source_url substring")
    p.add_argument("--main", action="store_true", help="Only main_content links")
    args = p.parse_args()

    clauses = []
    params: list[str] = []
    hay = "lower(coalesce(text,'')||' '||coalesce(url,'')||' '||coalesce(context,'')||' '||coalesce(categories,'')||' '||coalesce(section,'')||' '||coalesce(source_url,''))"
    for term in args.terms:
        clauses.append(f"{hay} LIKE ?")
        params.append(f"%{term.lower()}%")
    if args.source:
        clauses.append("source_url LIKE ?")
        params.append(f"%{args.source}%")
    if args.main:
        clauses.append("document_area='main_content'")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
      SELECT source_url, section, text, url, categories, context
      FROM links
      {where}
      ORDER BY document_area='main_content' DESC, source_url, ordinal
      LIMIT ?
    """
    params.append(str(args.limit))
    con = sqlite3.connect(DB)
    for source, section, text, url, cats, context in con.execute(sql, params):
        print(f"- {text or '(no text)'}")
        print(f"  url: {url}")
        print(f"  source: {source}")
        if section:
            print(f"  section: {section}")
        print(f"  categories: {cats}")
        if context and context != text:
            print(f"  context: {context[:300]}")


if __name__ == "__main__":
    main()
