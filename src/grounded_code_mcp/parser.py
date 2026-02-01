"""Document parsing using Docling."""

from __future__ import annotations

import logging
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
    ) -> None:
        """Initialize the parser.

        Args:
            enable_ocr: Whether to enable OCR for scanned documents.
            enable_table_extraction: Whether to extract tables from documents.
        """
        self._converter: DocumentConverter | None = None
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction

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

        # For plain markdown, just read the file directly
        if file_type == "md":
            return self._parse_markdown(path)

        # Use Docling for other formats
        return self._parse_with_docling(path, file_type)

    def _parse_markdown(self, path: Path) -> ParsedDocument:
        """Parse a markdown file directly.

        Args:
            path: Path to the markdown file.

        Returns:
            ParsedDocument with the file contents.
        """
        content = path.read_text(encoding="utf-8")

        # Try to extract title from first heading
        title = None
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ParsedDocument(
            path=path,
            content=content,
            title=title,
            file_type="md",
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
