"""Code-aware semantic chunking for documents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from grounded_code_mcp.config import ChunkingSettings


class Segment(TypedDict):
    """A segment extracted from document content."""

    type: str
    content: str
    start: int
    language: str | None


@dataclass
class Chunk:
    """A chunk of document content."""

    chunk_id: str
    content: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    heading_context: list[str] = field(default_factory=list)
    is_code: bool = False
    code_language: str | None = None
    is_table: bool = False
    source_path: str = ""

    @property
    def char_count(self) -> int:
        """Return the character count of the content."""
        return len(self.content)


def generate_chunk_id(source_path: str, index: int) -> str:
    """Generate a unique chunk ID as a valid UUID.

    Args:
        source_path: Path to the source document.
        index: Chunk index within the document.

    Returns:
        Unique chunk identifier as a UUID string.

    Note:
        Qdrant local client requires UUIDs for point IDs.
        The source_path and index are encoded in the UUID namespace.
    """
    # Generate a UUID5 based on source path and index for determinism,
    # but add randomness to handle re-ingestion
    _ = (source_path, index)  # Unused but documented for context
    return str(uuid.uuid4())


# Regex patterns
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
TABLE_PATTERN = re.compile(
    r"^\|.+\|\s*\n\|[-:\s|]+\|\s*\n(\|.+\|\s*\n)+",
    re.MULTILINE,
)

# Common function/method boundary patterns for splitting large code blocks
FUNCTION_PATTERNS = {
    "python": re.compile(r"^(?:async\s+)?def\s+\w+", re.MULTILINE),
    "javascript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+\w+|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^(?:export\s+)?(?:async\s+)?function\s+\w+|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>",
        re.MULTILINE,
    ),
    "java": re.compile(
        r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+\w+\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{",
        re.MULTILINE,
    ),
    "csharp": re.compile(
        r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(?:\w+\s+)+\w+\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    ),
    "go": re.compile(r"^func\s+(?:\([^)]+\)\s*)?\w+", re.MULTILINE),
    "rust": re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+\w+", re.MULTILINE),
    "php": re.compile(
        r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+\w+",
        re.MULTILINE,
    ),
}


@dataclass
class ChunkingContext:
    """Context maintained during chunking."""

    heading_stack: list[tuple[int, str]] = field(default_factory=list)
    current_index: int = 0
    source_path: str = ""

    def update_headings(self, level: int, text: str) -> None:
        """Update heading stack with a new heading.

        Args:
            level: Heading level (1-6).
            text: Heading text.
        """
        # Remove headings at same or lower level
        self.heading_stack = [
            (lvl, txt) for lvl, txt in self.heading_stack if lvl < level
        ]
        self.heading_stack.append((level, text))

    def get_heading_context(self) -> list[str]:
        """Get current heading context as list of strings."""
        return [text for _, text in self.heading_stack]

    def next_chunk_id(self) -> str:
        """Generate next chunk ID and increment index."""
        chunk_id = generate_chunk_id(self.source_path, self.current_index)
        self.current_index += 1
        return chunk_id


class DocumentChunker:
    """Code-aware semantic document chunker."""

    def __init__(
        self,
        text_chunk_size: int = 1000,
        text_chunk_max_size: int = 1500,
        text_chunk_overlap: int = 200,
        max_code_chunk_size: int = 3000,
    ) -> None:
        """Initialize the chunker.

        Args:
            text_chunk_size: Target size for text chunks.
            text_chunk_max_size: Maximum size for text chunks.
            text_chunk_overlap: Overlap between text chunks.
            max_code_chunk_size: Maximum size for code blocks before splitting.
        """
        self.text_chunk_size = text_chunk_size
        self.text_chunk_max_size = text_chunk_max_size
        self.text_chunk_overlap = text_chunk_overlap
        self.max_code_chunk_size = max_code_chunk_size

    @classmethod
    def from_settings(cls, settings: ChunkingSettings) -> DocumentChunker:
        """Create chunker from settings.

        Args:
            settings: Chunking settings.

        Returns:
            Configured DocumentChunker.
        """
        return cls(
            text_chunk_size=settings.text_chunk_size,
            text_chunk_max_size=settings.text_chunk_max_size,
            text_chunk_overlap=settings.text_chunk_overlap,
            max_code_chunk_size=settings.max_code_chunk_size,
        )

    def chunk(self, content: str, source_path: str = "") -> list[Chunk]:
        """Chunk document content.

        Args:
            content: Document content in markdown.
            source_path: Path to source document.

        Returns:
            List of chunks.
        """
        if not content.strip():
            return []

        ctx = ChunkingContext(source_path=source_path)
        chunks: list[Chunk] = []

        # Extract special blocks first (code blocks, tables)
        segments = self._extract_segments(content)

        for segment in segments:
            if segment["type"] == "code":
                chunks.extend(
                    self._chunk_code_block(
                        segment["content"],
                        segment["language"],
                        segment["start"],
                        ctx,
                    )
                )
            elif segment["type"] == "table":
                chunks.append(
                    self._create_table_chunk(
                        segment["content"],
                        segment["start"],
                        ctx,
                    )
                )
            else:  # text
                chunks.extend(
                    self._chunk_text(
                        segment["content"],
                        segment["start"],
                        ctx,
                    )
                )

        return chunks

    def _extract_segments(self, content: str) -> list[Segment]:
        """Extract code blocks, tables, and text segments.

        Args:
            content: Full document content.

        Returns:
            List of segments with type, content, and position.
        """
        segments: list[Segment] = []
        last_end = 0

        # Find all code blocks and tables
        special_blocks: list[tuple[int, int, str, str | None, str]] = []

        # Code blocks
        for match in CODE_BLOCK_PATTERN.finditer(content):
            special_blocks.append((
                match.start(),
                match.end(),
                "code",
                match.group(1) or None,  # language
                match.group(2),  # code content
            ))

        # Tables
        for match in TABLE_PATTERN.finditer(content):
            special_blocks.append((
                match.start(),
                match.end(),
                "table",
                None,
                match.group(0),
            ))

        # Sort by start position
        special_blocks.sort(key=lambda x: x[0])

        # Build segments
        for start, end, block_type, lang, block_content in special_blocks:
            # Add text before this block
            if start > last_end:
                text = content[last_end:start]
                if text.strip():
                    segments.append({
                        "type": "text",
                        "content": text,
                        "start": last_end,
                        "language": None,
                    })

            # Add the special block
            segments.append({
                "type": block_type,
                "content": block_content,
                "start": start,
                "language": lang,
            })
            last_end = end

        # Add remaining text
        if last_end < len(content):
            text = content[last_end:]
            if text.strip():
                segments.append({
                    "type": "text",
                    "content": text,
                    "start": last_end,
                    "language": None,
                })

        return segments

    def _chunk_code_block(
        self,
        code: str,
        language: str | None,
        start_char: int,
        ctx: ChunkingContext,
    ) -> list[Chunk]:
        """Chunk a code block.

        Code blocks are atomic up to max_code_chunk_size.
        Larger blocks are split on function boundaries.

        Args:
            code: Code content.
            language: Programming language.
            start_char: Start position in document.
            ctx: Chunking context.

        Returns:
            List of code chunks.
        """
        code = code.strip()
        if not code:
            return []

        # If small enough, keep as single chunk
        if len(code) <= self.max_code_chunk_size:
            return [
                Chunk(
                    chunk_id=ctx.next_chunk_id(),
                    content=code,
                    chunk_index=ctx.current_index - 1,
                    start_char=start_char,
                    end_char=start_char + len(code),
                    heading_context=ctx.get_heading_context(),
                    is_code=True,
                    code_language=language,
                    source_path=ctx.source_path,
                )
            ]

        # Try to split on function boundaries
        chunks = self._split_code_on_functions(code, language, start_char, ctx)
        if chunks:
            return chunks

        # Fallback: split on double newlines
        return self._split_code_on_newlines(code, language, start_char, ctx)

    def _split_code_on_functions(
        self,
        code: str,
        language: str | None,
        start_char: int,
        ctx: ChunkingContext,
    ) -> list[Chunk]:
        """Split code on function boundaries.

        Args:
            code: Code content.
            language: Programming language.
            start_char: Start position.
            ctx: Chunking context.

        Returns:
            List of chunks, or empty if no function boundaries found.
        """
        if not language or language.lower() not in FUNCTION_PATTERNS:
            return []

        pattern = FUNCTION_PATTERNS[language.lower()]
        matches = list(pattern.finditer(code))

        if len(matches) < 2:
            return []

        chunks: list[Chunk] = []
        positions = [m.start() for m in matches] + [len(code)]

        for i in range(len(positions) - 1):
            segment = code[positions[i] : positions[i + 1]].strip()
            if segment:
                chunks.append(
                    Chunk(
                        chunk_id=ctx.next_chunk_id(),
                        content=segment,
                        chunk_index=ctx.current_index - 1,
                        start_char=start_char + positions[i],
                        end_char=start_char + positions[i + 1],
                        heading_context=ctx.get_heading_context(),
                        is_code=True,
                        code_language=language,
                        source_path=ctx.source_path,
                    )
                )

        return chunks

    def _split_code_on_newlines(
        self,
        code: str,
        language: str | None,
        start_char: int,
        ctx: ChunkingContext,
    ) -> list[Chunk]:
        """Split code on double newlines as fallback.

        Args:
            code: Code content.
            language: Programming language.
            start_char: Start position.
            ctx: Chunking context.

        Returns:
            List of chunks.
        """
        sections = re.split(r"\n\n+", code)
        chunks: list[Chunk] = []
        current_chunk = ""
        current_start = start_char

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current_chunk) + len(section) + 2 <= self.max_code_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
            else:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            chunk_id=ctx.next_chunk_id(),
                            content=current_chunk,
                            chunk_index=ctx.current_index - 1,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            heading_context=ctx.get_heading_context(),
                            is_code=True,
                            code_language=language,
                            source_path=ctx.source_path,
                        )
                    )
                current_chunk = section
                current_start = start_char + code.find(section)

        if current_chunk:
            chunks.append(
                Chunk(
                    chunk_id=ctx.next_chunk_id(),
                    content=current_chunk,
                    chunk_index=ctx.current_index - 1,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    heading_context=ctx.get_heading_context(),
                    is_code=True,
                    code_language=language,
                    source_path=ctx.source_path,
                )
            )

        return chunks

    def _create_table_chunk(
        self,
        table: str,
        start_char: int,
        ctx: ChunkingContext,
    ) -> Chunk:
        """Create a chunk for a table (tables are atomic).

        Args:
            table: Table content.
            start_char: Start position.
            ctx: Chunking context.

        Returns:
            Table chunk.
        """
        return Chunk(
            chunk_id=ctx.next_chunk_id(),
            content=table.strip(),
            chunk_index=ctx.current_index - 1,
            start_char=start_char,
            end_char=start_char + len(table),
            heading_context=ctx.get_heading_context(),
            is_table=True,
            source_path=ctx.source_path,
        )

    def _chunk_text(
        self,
        text: str,
        start_char: int,
        ctx: ChunkingContext,
    ) -> list[Chunk]:
        """Chunk regular text content.

        Args:
            text: Text content.
            start_char: Start position.
            ctx: Chunking context.

        Returns:
            List of text chunks.
        """
        # Update heading context as we process
        for match in HEADING_PATTERN.finditer(text):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            ctx.update_headings(level, heading_text)

        # Split into paragraphs
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[Chunk] = []
        current_chunk = ""
        current_start = start_char

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for headings to update context
            heading_match = HEADING_PATTERN.match(para)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                ctx.update_headings(level, heading_text)

            # If adding this paragraph exceeds max size, flush current chunk
            if (
                current_chunk
                and len(current_chunk) + len(para) + 2 > self.text_chunk_max_size
            ):
                chunks.append(
                    Chunk(
                        chunk_id=ctx.next_chunk_id(),
                        content=current_chunk,
                        chunk_index=ctx.current_index - 1,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        heading_context=ctx.get_heading_context(),
                        source_path=ctx.source_path,
                    )
                )
                # Apply overlap by keeping last part of previous chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para if overlap_text else para
                current_start = start_char + text.find(para)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

            # If we've reached target size, consider flushing
            if len(current_chunk) >= self.text_chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=ctx.next_chunk_id(),
                        content=current_chunk,
                        chunk_index=ctx.current_index - 1,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        heading_context=ctx.get_heading_context(),
                        source_path=ctx.source_path,
                    )
                )
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text
                current_start = start_char + text.find(para) + len(para)

        # Flush remaining content
        if current_chunk.strip():
            chunks.append(
                Chunk(
                    chunk_id=ctx.next_chunk_id(),
                    content=current_chunk.strip(),
                    chunk_index=ctx.current_index - 1,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    heading_context=ctx.get_heading_context(),
                    source_path=ctx.source_path,
                )
            )

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk.

        Args:
            text: Chunk text.

        Returns:
            Overlap portion to prepend to next chunk.
        """
        if len(text) <= self.text_chunk_overlap:
            return ""

        # Try to break at sentence boundary
        overlap_region = text[-self.text_chunk_overlap :]
        sentence_end = max(
            overlap_region.rfind(". "),
            overlap_region.rfind("! "),
            overlap_region.rfind("? "),
        )

        if sentence_end > 0:
            return overlap_region[sentence_end + 2 :]

        # Fallback: break at word boundary
        word_break = overlap_region.rfind(" ")
        if word_break > 0:
            return overlap_region[word_break + 1 :]

        return overlap_region
