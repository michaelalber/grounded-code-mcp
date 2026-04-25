#!/usr/bin/env python3
"""
Convert eCFR bulk XML exports to clean Markdown files for RAG ingestion.

Produces one .md file per CFR Part, filtered to only the parts you request.

Usage:
    python scripts/ecfr_xml_to_md.py \\
        --input path/to/ecfr.xml \\
        --parts 730 731 732 \\
        --output sources/gov/ear \\
        --title "Export Administration Regulations (15 CFR)"

    # Convert all parts in file (omit --parts):
    python scripts/ecfr_xml_to_md.py \\
        --input path/to/10cfr712.xml \\
        --output sources/gov/cfr \\
        --title "10 CFR Part 712 — Human Reliability Program"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


# ── XML tag → heading level ────────────────────────────────────────────────
_DIV_LEVELS: dict[str, int] = {
    "DIV1": 1,  # Title
    "DIV2": 1,  # Subtitle
    "DIV3": 2,  # Chapter
    "DIV4": 2,  # Subchapter
    "DIV5": 1,  # Part  ← reset to H1 so each file feels self-contained
    "DIV6": 2,  # Subpart
    "DIV7": 3,  # Subject group
    "DIV8": 4,  # Section
    "DIV9": 5,  # Paragraph group
}


def _inline_text(element: ET.Element) -> str:
    """Recursively extract text, preserving inline markup as plain text."""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        tag = child.tag
        if tag in ("I", "SU", "SUB", "SUP"):
            parts.append(f"*{_inline_text(child)}*")
        elif tag in ("B", "BOLD"):
            parts.append(f"**{_inline_text(child)}**")
        elif tag == "E":
            t_attr = child.get("T", "")
            inner = _inline_text(child)
            parts.append(f"**{inner}**" if t_attr in ("03", "04") else f"*{inner}*")
        else:
            parts.append(_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _clean(text: str) -> str:
    """Collapse whitespace runs."""
    return re.sub(r"\s+", " ", text).strip()


def _convert_element(el: ET.Element, lines: list[str]) -> None:
    """Recursively walk an element tree, appending markdown lines."""
    tag = el.tag

    if tag in _DIV_LEVELS:
        level = _DIV_LEVELS[tag]
        head_el = el.find("HEAD")
        if head_el is not None:
            heading = _clean(_inline_text(head_el))
            lines.append(f"\n{'#' * level} {heading}\n")
        for child in el:
            if child.tag != "HEAD":
                _convert_element(child, lines)
        return

    if tag == "HEAD":
        return  # already handled by parent DIV

    if tag == "P":
        text = _clean(_inline_text(el))
        if text:
            lines.append(f"\n{text}\n")
        return

    if tag in ("AUTH", "SOURCE"):
        hed = el.find("HED")
        ps = el.find("PSPACE")
        label = _clean(hed.text) if hed is not None and hed.text else tag
        content = _clean(_inline_text(ps)) if ps is not None else ""
        if content:
            lines.append(f"\n> **{label}** {content}\n")
        return

    if tag == "CITA":
        text = _clean(_inline_text(el))
        if text:
            lines.append(f"\n*{text}*\n")
        return

    if tag == "EDNOTE":
        hed = el.find("HED")
        ps = el.find("PSPACE")
        label = _clean(hed.text) if hed is not None and hed.text else "Note"
        content = _clean(_inline_text(ps)) if ps is not None else _clean(_inline_text(el))
        if content:
            lines.append(f"\n> **{label}** {content}\n")
        return

    if tag in ("FP", "FP1", "FP2", "FP-1", "FP-2"):
        text = _clean(_inline_text(el))
        if text:
            lines.append(f"\n{text}\n")
        return

    if tag in ("GPOTABLE", "TTITLE", "TDESC", "TDATA"):
        # Tables: emit as plain text rows
        text = _clean(_inline_text(el))
        if text:
            lines.append(f"\n{text}\n")
        return

    # Fallback: recurse into children
    for child in el:
        _convert_element(child, lines)


def _part_number(div5: ET.Element) -> str | None:
    return div5.get("N")


def find_parts(root: ET.Element, wanted: set[str] | None) -> list[ET.Element]:
    """Return DIV5 (Part) elements matching wanted part numbers (or all)."""
    parts: list[ET.Element] = []
    for div5 in root.iter("DIV5"):
        if div5.get("TYPE") == "PART":
            n = _part_number(div5)
            if wanted is None or (n is not None and n in wanted):
                parts.append(div5)
    return parts


def convert_part(div5: ET.Element, title_prefix: str) -> tuple[str, str]:
    """Convert a DIV5 Part to (filename_stem, markdown_text)."""
    n = _part_number(div5) or "unknown"
    head_el = div5.find("HEAD")
    part_title = _clean(_inline_text(head_el)) if head_el is not None else f"Part {n}"

    lines: list[str] = [f"# {part_title}\n"]
    if title_prefix:
        lines.insert(0, f"<!-- Source: {title_prefix} -->\n")

    for child in div5:
        if child.tag != "HEAD":
            _convert_element(child, lines)

    return f"cfr-part-{n}", "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert eCFR XML to Markdown for RAG ingestion")
    parser.add_argument("--input", required=True, help="Path to eCFR bulk XML file")
    parser.add_argument("--parts", nargs="*", help="CFR Part numbers to extract (omit for all)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--title", default="", help="Human-readable title prefix for comments")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted: set[str] | None = set(args.parts) if args.parts else None

    print(f"Parsing {input_path} ({input_path.stat().st_size // 1024} KB)...")
    tree = ET.parse(input_path)
    root = tree.getroot()

    parts = find_parts(root, wanted)
    if not parts:
        print("ERROR: no matching parts found. Check --parts values.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parts)} part(s). Converting...")
    for div5 in parts:
        stem, md = convert_part(div5, args.title)
        out_file = output_dir / f"{stem}.md"
        out_file.write_text(md, encoding="utf-8")
        size_kb = len(md) // 1024
        print(f"  Wrote {out_file.name} ({size_kb} KB)")

    print(f"Done. {len(parts)} file(s) in {output_dir}/")


if __name__ == "__main__":
    main()
