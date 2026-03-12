#!/usr/bin/env python3
# <AI-Generated START>
"""Convert a large PDF to markdown files using Docling with pypdf page-range batching.

Splits the input PDF into temporary per-batch PDFs, converts each with Docling
(preserving headings, tables, lists, code blocks), then writes one .md file per batch.
Temporary files are always cleaned up, even on failure.

Requires: docling, pypdf

Usage:
    python scripts/split_pdf_to_markdown.py <pdf_path> <output_dir> [--pages-per-file N]

Example:
    python scripts/split_pdf_to_markdown.py \\
        sources/dotnet/Telerik.UI.for.Blazor.13.0.0.pdf \\
        sources/dotnet/telerik-blazor \\
        --pages-per-file 150
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def get_page_count(pdf_path: Path) -> int:
    """Return the total page count using pypdf.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Total number of pages.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def extract_page_range(pdf_path: Path, first: int, last: int, dest: Path) -> None:
    """Write pages [first, last] (1-based, inclusive) to a new PDF at dest.

    Args:
        pdf_path: Source PDF path.
        first: First page number (1-based).
        last: Last page number (1-based, inclusive).
        dest: Destination path for the extracted PDF.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    for page_index in range(first - 1, last):  # convert to 0-based
        writer.add_page(reader.pages[page_index])

    with dest.open("wb") as fh:
        writer.write(fh)


def convert_with_docling(pdf_path: Path) -> str:
    """Convert a PDF to markdown using Docling.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Markdown string.

    Raises:
        ImportError: If docling is not installed.
        RuntimeError: If conversion fails.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError("docling is required: pip install docling") from exc

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def split_pdf(pdf_path: Path, output_dir: Path, pages_per_batch: int) -> None:
    """Split a large PDF into batches and convert each with Docling.

    Args:
        pdf_path: Source PDF path.
        output_dir: Directory to write markdown files into.
        pages_per_batch: Number of pages per batch.
    """
    total = get_page_count(pdf_path)
    print(f"{pdf_path.name}: {total} pages → batches of {pages_per_batch}")

    output_dir.mkdir(parents=True, exist_ok=True)

    batch = 1
    first = 1
    while first <= total:
        last = min(first + pages_per_batch - 1, total)
        print(f"  Batch {batch:03d}: pages {first}–{last} … ", end="", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            extract_page_range(pdf_path, first, last, tmp_path)
            markdown = convert_with_docling(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        out_file = output_dir / f"{batch:03d}-pages-{first:05d}-{last:05d}.md"
        out_file.write_text(markdown, encoding="utf-8")
        print(f"{len(markdown):,} chars → {out_file.name}")

        first = last + 1
        batch += 1

    print(f"\nDone — {batch - 1} file(s) written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pages-per-file", type=int, default=150)
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: {args.pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    split_pdf(args.pdf_path, args.output_dir, args.pages_per_file)


if __name__ == "__main__":
    main()
# <AI-Generated END>
