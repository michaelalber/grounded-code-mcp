"""Document parsing using Docling."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".mdx",
    ".rst",
    ".txt",
    ".asciidoc",
    ".adoc",
    ".epub",
    ".xlsx",
    ".xls",
}


@dataclass
class ParsedDocument:
    """Result of parsing a document."""

    path: Path
    content: str
    title: str | None = None
    page_count: int | None = None
    file_type: str = ""
    metadata: dict[str, str | int | float | bool | None] | None = None

    @property
    def is_empty(self) -> bool:
        """Check if the document has no content."""
        return not self.content or not self.content.strip()


class DocumentParseError(Exception):
    """Error during document parsing."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"Failed to parse {path}: {message}")


class UnsupportedFormatError(Exception):
    """Raised for unsupported file formats."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Unsupported file format: {path.suffix}")


def is_supported_format(path: Path) -> bool:
    """Check if a file format is supported.

    Args:
        path: Path to check.

    Returns:
        True if the format is supported.
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_file_type(path: Path) -> str:
    """Get the file type from a path.

    Args:
        path: Path to the file.

    Returns:
        File type string (e.g., "pdf", "docx", "md").
    """
    suffix = path.suffix.lower()
    # Normalize some extensions
    type_map = {
        ".markdown": "md",
        ".mdx": "md",
        ".adoc": "asciidoc",
        ".htm": "html",
        ".doc": "docx",
        ".ppt": "pptx",
        ".xls": "xlsx",
    }
    return type_map.get(suffix, suffix.lstrip("."))


class DocumentParser:
    """Parser for converting documents to markdown using Docling."""

    def __init__(
        self,
        *,
        enable_ocr: bool = True,
        enable_table_extraction: bool = True,
        pdf_page_batch_size: int = 0,
    ) -> None:
        """Initialize the parser.

        Args:
            enable_ocr: Whether to enable OCR for scanned documents.
            enable_table_extraction: Whether to extract tables from documents.
            pdf_page_batch_size: Split PDFs into batches of this many pages before
                passing to Docling. Bounds peak memory for large files. 0 disables.
        """
        self._converter: DocumentConverter | None = None
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction
        self.pdf_page_batch_size = pdf_page_batch_size

    def _get_converter(self) -> DocumentConverter:
        """Lazy-load the Docling converter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a document to markdown.

        Args:
            path: Path to the document.

        Returns:
            ParsedDocument with markdown content.

        Raises:
            UnsupportedFormatError: If the format is not supported.
            DocumentParseError: If parsing fails.
            FileNotFoundError: If the file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not is_supported_format(path):
            raise UnsupportedFormatError(path)

        file_type = get_file_type(path)

        # For plain text formats, read the file directly
        if file_type in ("md", "asciidoc", "rst", "txt"):
            return self._parse_plaintext(path, file_type)

        # Large-PDF path: split into page batches before handing to Docling
        if file_type == "pdf" and self.pdf_page_batch_size > 0:
            return self._parse_pdf_in_batches(path)

        # Use Docling for other formats
        return self._parse_with_docling(path, file_type)

    def _parse_plaintext(self, path: Path, file_type: str) -> ParsedDocument:
        """Parse a plain text file directly (markdown, asciidoc, rst, txt, etc.).

        Args:
            path: Path to the file.
            file_type: Normalized file type string.

        Returns:
            ParsedDocument with the file contents.
        """
        content = path.read_text(encoding="utf-8")

        title = None
        lines = content.split("\n")

        if file_type in ("md",):
            for line in lines:
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        elif file_type == "asciidoc":
            for line in lines:
                line = line.strip()
                if line.startswith("= "):
                    title = line[2:].strip()
                    break
        elif file_type == "rst":
            # RST titles: a text line followed by an underline of ===, ---, ~~~, etc.
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and i + 1 < len(lines):
                    underline = lines[i + 1].strip()
                    if underline and len(underline) >= len(stripped) and underline == underline[0] * len(underline) and underline[0] in "=-~^#+*":
                        title = stripped
                        break
        # txt: no title extraction

        return ParsedDocument(
            path=path,
            content=content,
            title=title,
            file_type=file_type,
        )

    def _parse_with_docling(self, path: Path, file_type: str) -> ParsedDocument:
        """Parse a document using Docling.

        Args:
            path: Path to the document.
            file_type: The file type.

        Returns:
            ParsedDocument with markdown content.

        Raises:
            DocumentParseError: If parsing fails.
        """
        try:
            converter = self._get_converter()
            result = converter.convert(str(path))

            # Export to markdown
            markdown_content = result.document.export_to_markdown()

            # Extract metadata
            title = None
            page_count = None

            # Try to get title from document metadata
            if hasattr(result.document, "title") and result.document.title:
                title = result.document.title
            elif (
                hasattr(result.document, "metadata")
                and result.document.metadata
                and hasattr(result.document.metadata, "title")
            ):
                title = result.document.metadata.title

            # Try to get page count
            if hasattr(result.document, "pages"):
                page_count = len(result.document.pages)

            return ParsedDocument(
                path=path,
                content=markdown_content,
                title=title,
                page_count=page_count,
                file_type=file_type,
            )

        except Exception as e:
            logger.exception("Failed to parse document: %s", path)
            raise DocumentParseError(path, str(e)) from e

    def _get_pdf_page_count(self, path: Path) -> int:
        """Return total page count of a PDF using pypdf.

        Args:
            path: Path to the PDF file.

        Returns:
            Total number of pages.
        """
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required for batched PDF parsing: pip install pypdf") from exc

        return len(PdfReader(str(path)).pages)

    def _split_pdf_batch_to_temp(self, path: Path, first: int, last: int, dest: Path) -> None:
        """Extract pages [first, last) (0-based) from path and write to dest.

        Args:
            path: Source PDF path.
            first: First page index (0-based, inclusive).
            last: Last page index (0-based, exclusive).
            dest: Destination path for the extracted PDF batch.
        """
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(path))
        writer = PdfWriter()
        for i in range(first, last):
            writer.add_page(reader.pages[i])
        with dest.open("wb") as fh:
            writer.write(fh)

    def _parse_pdf_in_batches(self, path: Path) -> ParsedDocument:
        """Parse a large PDF by splitting into page batches and running Docling on each.

        Splits the PDF into temporary files of ``pdf_page_batch_size`` pages,
        converts each with Docling, concatenates the markdown, and cleans up
        temp files even if conversion fails.

        Args:
            path: Path to the source PDF.

        Returns:
            ParsedDocument with concatenated markdown and total page count.

        Raises:
            DocumentParseError: If any batch fails to convert.
        """
        total = self._get_pdf_page_count(path)
        markdowns: list[str] = []
        first = 0

        while first < total:
            last = min(first + self.pdf_page_batch_size, total)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            try:
                self._split_pdf_batch_to_temp(path, first, last, tmp_path)
                batch = self._parse_with_docling(tmp_path, "pdf")
                markdowns.append(batch.content)
            finally:
                tmp_path.unlink(missing_ok=True)

            first = last

        return ParsedDocument(
            path=path,
            content="\n\n".join(markdowns),
            title=None,
            page_count=total,
            file_type="pdf",
        )

    def parse_many(
        self,
        paths: list[Path],
        *,
        skip_errors: bool = True,
    ) -> list[ParsedDocument]:
        """Parse multiple documents.

        Args:
            paths: List of paths to parse.
            skip_errors: If True, log errors and continue; if False, raise on first error.

        Returns:
            List of successfully parsed documents.
        """
        results: list[ParsedDocument] = []

        for path in paths:
            try:
                result = self.parse(path)
                if not result.is_empty:
                    results.append(result)
                else:
                    logger.warning("Skipping empty document: %s", path)
            except UnsupportedFormatError:
                logger.warning("Skipping unsupported format: %s", path)
                if not skip_errors:
                    raise
            except DocumentParseError as e:
                logger.error("Failed to parse %s: %s", path, e)
                if not skip_errors:
                    raise
            except FileNotFoundError:
                logger.error("File not found: %s", path)
                if not skip_errors:
                    raise

        return results


def scan_directory(
    directory: Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Scan a directory for supported documents.

    Args:
        directory: Directory to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        List of paths to supported documents.
    """
    if not directory.exists():
        return []

    pattern = "**/*" if recursive else "*"
    paths = []

    for path in directory.glob(pattern):
        if path.is_file() and is_supported_format(path):
            paths.append(path)

    return sorted(paths)
