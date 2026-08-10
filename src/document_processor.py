"""Document ingestion and structure-aware chunking for the research-paper RAG system.

The corpus consists of Marker-extracted research-paper ``.txt`` files.  Marker
already preserves useful structure (Markdown headings, tables, figure captions,
and page markers), so this module cleans those artifacts and turns the papers
into retrieval-ready chunks while preserving provenance.

Pipeline
--------
TextFileLoader
    -> loads raw paper files and basic paper metadata
MarkerCleaner
    -> removes Marker-only artifacts while preserving useful Markdown
SectionParser
    -> detects section headings and page boundaries
StructureAwareChunker
    -> creates paragraph/section-aware chunks with bounded overlap
DocumentProcessor
    -> orchestrates the complete ingestion process

The public ``DocumentChunk`` / ``TextFileLoader`` / ``DocumentChunker`` names
are retained for compatibility with the existing pipeline while the new
``DocumentProcessor`` is the preferred interface going forward.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Data models
@dataclass
class RawDocument:
    """A loaded source document with basic provenance."""

    text: str
    source: str
    path: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Section:
    """A logical section of a research paper."""

    title: str
    text: str
    number: str | None = None
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class DocumentChunk:
    """A retrieval-ready chunk with provenance metadata.

    ``text`` is the content sent to the embedding model.  ``metadata`` keeps
    information that should not be embedded, such as page numbers and section
    names, but is needed later for source attribution.
    """

    text: str
    source: str
    chunk_id: int
    metadata: dict[str, str] = field(default_factory=dict)

# Loading
class TextFileLoader:
    """Load Marker-extracted ``.txt`` research papers from a directory."""

    def load_documents(self, data_dir: Path) -> list[RawDocument]:
        """Load all non-empty ``.txt`` files with basic metadata."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory: {data_dir}")

        documents: list[RawDocument] = []

        for path in sorted(data_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                logger.error("Failed to read %s: %s", path, exc)
                continue

            if not text:
                logger.warning("Skipping empty document: %s", path.name)
                continue

            metadata = _extract_paper_metadata(text, path.stem)
            documents.append(
                RawDocument(
                    text=text,
                    source=path.stem,
                    path=str(path),
                    metadata=metadata,
                )
            )
            logger.info(
                "Loaded '%s' — %d characters", path.name, len(text)
            )

        logger.info("Loaded %d document(s) from %s", len(documents), data_dir)
        return documents

    def load(self, data_dir: Path) -> list[tuple[str, str]]:
        """Backward-compatible loader returning ``(text, source)`` tuples.

        The existing pipeline uses this interface.  New code should prefer
        ``load_documents`` so metadata is retained.
        """
        return [
            (document.text, document.source)
            for document in self.load_documents(data_dir)
        ]

# Marker cleanup
class MarkerCleaner:
    """Clean recurring Marker extraction artifacts.

    The cleaner deliberately does *not* strip Markdown headings, tables, or
    figure captions because those structures are useful for retrieval.
    """

    _PAGE_MARKER_RE = re.compile(
        r"<span\s+id=[\"']page-(\d+)-\d+[\"']\s*>\s*</span>",
        flags=re.IGNORECASE,
    )
    _HTML_TAG_RE = re.compile(r"</?(?!sup\b|sub\b|br\b)[a-zA-Z][^>]*>")
    _IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    _LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
    _REFERENCE_LINK_RE = re.compile(r"\(#page-\d+-\d+\)")
    _MULTISPACE_RE = re.compile(r"[ \t]{2,}")
    _MULTINEWLINE_RE = re.compile(r"\n{3,}")

    def clean(self, text: str) -> tuple[str, dict[int, int]]:
        """Return cleaned text and a map of source page markers.

        Page markers are removed from the text but converted into a mapping
        from character position to page number.  The parser uses this mapping
        to attach page provenance to chunks.
        """
        page_positions: dict[int, int] = {}

        def record_page(match: re.Match[str]) -> str:
            page = int(match.group(1))
            page_positions[match.start()] = page
            return f"\n<!-- PAGE:{page} -->\n"

        text = self._PAGE_MARKER_RE.sub(record_page, text)

        # Keep image captions, but remove the image syntax itself.
        text = self._IMAGE_RE.sub("", text)

        # Convert links to their visible text while retaining plain URLs.
        text = self._LINK_RE.sub(r"\1", text)
        text = self._REFERENCE_LINK_RE.sub("", text)

        # Decode entities introduced by HTML extraction.
        text = html.unescape(text)

        # Remove ordinary HTML tags but retain <sup>/<sub>/<br> because they
        # can carry useful mathematical structure.
        text = self._HTML_TAG_RE.sub("", text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

        # Normalize a few common extraction artifacts.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = self._MULTISPACE_RE.sub(" ", text)
        text = self._MULTINEWLINE_RE.sub("\n\n", text)

        return text.strip(), page_positions

# Section parsing
@dataclass
class _ParsedBlock:
    """Internal block representation used before chunking."""

    text: str
    section: str
    section_number: str | None
    level: int
    page_start: int | None
    page_end: int | None
    content_type: str = "text"


class SectionParser:
    """Parse Markdown-style paper sections and page boundaries."""

    _HEADING_RE = re.compile(
        r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$"
    )
    _PAGE_RE = re.compile(r"^\s*<!--\s*PAGE:(\d+)\s*-->\s*$")
    _NUMBERED_SECTION_RE = re.compile(
        r"^(?P<number>(?:\d+\.)*\d+)\s+(?P<title>.+)$"
    )
    _REFERENCES_RE = re.compile(
        r"^(?:references|bibliography)\s*$", re.IGNORECASE
    )

    def parse(self, text: str) -> list[Section]:
        """Convert cleaned text into logical sections.

        Content before the first heading is retained as ``Front Matter``.
        """
        lines = text.splitlines()
        sections: list[Section] = []

        current_title = "Front Matter"
        current_number: str | None = None
        current_level = 1
        current_lines: list[str] = []
        current_page_start: int | None = None
        current_page_end: int | None = None

        def flush() -> None:
            nonlocal current_lines
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    Section(
                        title=current_title,
                        text=content,
                        number=current_number,
                        level=current_level,
                        page_start=current_page_start,
                        page_end=current_page_end,
                    )
                )
            current_lines = []

        for line in lines:
            page_match = self._PAGE_RE.match(line)
            if page_match:
                page = int(page_match.group(1))
                if current_page_start is None:
                    current_page_start = page
                current_page_end = page
                continue

            heading_match = self._HEADING_RE.match(line)
            if heading_match:
                flush()
                title = heading_match.group("title").strip()
                level = len(heading_match.group("marks"))
                number, clean_title = self._split_section_number(title)

                current_title = clean_title
                current_number = number
                current_level = level
                current_page_start = current_page_end
                continue

            current_lines.append(line)

        flush()
        return sections

    def _split_section_number(
        self, title: str
    ) -> tuple[str | None, str]:
        match = self._NUMBERED_SECTION_RE.match(title)
        if match:
            return match.group("number"), match.group("title").strip()
        return None, title

# Structure-aware chunking
class StructureAwareChunker:
    """Create chunks from sections without cutting paragraphs unnecessarily.

    ``target_chars`` is a target rather than a hard limit.  A block that is
    already larger than ``max_chars`` is split at sentence/word boundaries.
    Tables are kept together whenever possible.
    """

    _TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    _TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
    _FIGURE_CAPTION_RE = re.compile(
        r"^\s*(?:Figure|Fig\.|Table)\s+\d+[:.]",
        re.IGNORECASE,
    )
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def __init__(
        self,
        target_chars: int = 1800,
        max_chars: int = 2600,
        overlap_chars: int = 250,
        exclude_references: bool = True,
    ) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if max_chars < target_chars:
            raise ValueError("max_chars must be >= target_chars")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be >= 0 and < target_chars")

        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.exclude_references = exclude_references

    def chunk_sections(
        self,
        sections: Iterable[Section],
        source: str,
        base_metadata: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk parsed sections while preserving section/page provenance."""
        chunks: list[DocumentChunk] = []
        chunk_id = 0
        base_metadata = dict(base_metadata or {})

        for section in sections:
            if self.exclude_references and self._is_references(section):
                logger.debug("Excluded References section from '%s'", source)
                continue

            blocks = self._split_into_blocks(section.text)

            for block_text, content_type in blocks:
                for piece in self._pack_block(block_text):
                    if not piece.strip():
                        continue

                    metadata = {
                        **base_metadata,
                        "section": section.title,
                        "section_level": str(section.level),
                        "content_type": content_type,
                    }

                    if section.number:
                        metadata["section_number"] = section.number
                    if section.page_start is not None:
                        metadata["page_start"] = str(section.page_start)
                    if section.page_end is not None:
                        metadata["page_end"] = str(section.page_end)

                    chunks.append(
                        DocumentChunk(
                            text=piece.strip(),
                            source=source,
                            chunk_id=chunk_id,
                            metadata=metadata,
                        )
                    )
                    chunk_id += 1

        logger.info("'%s' → %d structure-aware chunks", source, len(chunks))
        return chunks

    def _is_references(self, section: Section) -> bool:
        return section.title.strip().lower() in {
            "references",
            "bibliography",
        }

    def _split_into_blocks(self, text: str) -> list[tuple[str, str]]:
        """Split section text into paragraphs, tables, and captions."""
        lines = text.splitlines()
        blocks: list[tuple[str, str]] = []
        current: list[str] = []
        in_table = False

        def flush_text() -> None:
            nonlocal current
            content = "\n".join(current).strip()
            if content:
                blocks.append((content, "text"))
            current = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Markdown table: header + separator + subsequent table rows.
            if (
                i + 1 < len(lines)
                and self._TABLE_ROW_RE.match(line)
                and self._TABLE_SEPARATOR_RE.match(lines[i + 1])
            ):
                flush_text()
                table = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and self._TABLE_ROW_RE.match(lines[i]):
                    table.append(lines[i])
                    i += 1
                blocks.append(("\n".join(table).strip(), "table"))
                continue

            if not line.strip():
                flush_text()
                i += 1
                continue

            if self._FIGURE_CAPTION_RE.match(line):
                flush_text()
                blocks.append((line.strip(), "caption"))
                i += 1
                continue

            current.append(line)
            i += 1

        flush_text()
        return blocks

    def _pack_block(self, block: str) -> list[str]:
        """Pack text into target-sized pieces using sentence boundaries."""
        block = block.strip()
        if len(block) <= self.max_chars:
            return [block]

        # Tables should not be arbitrarily split if possible.
        if self._TABLE_ROW_RE.match(block.splitlines()[0]):
            return self._split_hard(block)

        sentences = self._SENTENCE_RE.split(block)
        pieces: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = (
                sentence if not current else f"{current} {sentence}"
            )

            if len(candidate) <= self.target_chars:
                current = candidate
                continue

            if current:
                pieces.append(current)

            if len(sentence) <= self.max_chars:
                current = sentence
            else:
                pieces.extend(self._split_hard(sentence))
                current = ""

        if current:
            pieces.append(current)

        return self._add_overlap(pieces)

    def _split_hard(self, text: str) -> list[str]:
        """Fallback split for unusually long paragraphs/equations/tables."""
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            added_len = len(word) + (1 if current else 0)
            if current and current_len + added_len > self.max_chars:
                pieces.append(" ".join(current))
                current = []
                current_len = 0

            current.append(word)
            current_len += added_len

        if current:
            pieces.append(" ".join(current))

        return self._add_overlap(pieces)

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        """Carry a small tail of the previous piece into the next piece."""
        if self.overlap_chars == 0 or len(pieces) < 2:
            return pieces

        result = [pieces[0]]
        for piece in pieces[1:]:
            tail = result[-1][-self.overlap_chars:].strip()
            result.append(f"{tail}\n{piece}".strip() if tail else piece)

        return result

# High-level processor
class DocumentProcessor:
    """End-to-end processor for the research-paper corpus."""

    def __init__(
        self,
        target_chars: int = 1800,
        max_chars: int = 2600,
        overlap_chars: int = 250,
        exclude_references: bool = True,
    ) -> None:
        self.loader = TextFileLoader()
        self.cleaner = MarkerCleaner()
        self.parser = SectionParser()
        self.chunker = StructureAwareChunker(
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            exclude_references=exclude_references,
        )

    def process_directory(self, data_dir: Path) -> list[DocumentChunk]:
        """Process every paper in ``data_dir`` into retrieval-ready chunks."""
        documents = self.loader.load_documents(data_dir)

        if not documents:
            raise FileNotFoundError(
                f"No non-empty .txt documents found in '{data_dir}'."
            )

        all_chunks: list[DocumentChunk] = []

        for document in documents:
            cleaned_text, _ = self.cleaner.clean(document.text)
            sections = self.parser.parse(cleaned_text)

            chunks = self.chunker.chunk_sections(
                sections,
                source=document.source,
                base_metadata={
                    **document.metadata,
                    "source_file": Path(document.path).name,
                },
            )
            all_chunks.extend(chunks)

        logger.info(
            "Processed %d document(s) → %d total chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

# Backward-compatible wrapper
class DocumentChunker:
    """Compatibility wrapper around ``StructureAwareChunker``.

    New code should use ``DocumentProcessor``.  This class remains so the
    existing pipeline does not immediately break while we migrate it.
    """

    def __init__(
        self,
        chunk_size: int = 1800,
        chunk_overlap: int = 250,
    ) -> None:
        self._chunker = StructureAwareChunker(
            target_chars=chunk_size,
            max_chars=chunk_size,
            overlap_chars=min(chunk_overlap, max(0, chunk_size // 4)),
            exclude_references=False,
        )

    def chunk(self, text: str, source: str) -> list[DocumentChunk]:
        """Chunk raw text using the new structure-aware implementation."""
        cleaner = MarkerCleaner()
        parser = SectionParser()

        cleaned_text, _ = cleaner.clean(text)
        sections = parser.parse(cleaned_text)

        return self._chunker.chunk_sections(sections, source)

# Metadata helpers
def _extract_paper_metadata(text: str, source: str) -> dict[str, str]:
    """Extract lightweight metadata from the beginning of a paper.

    This intentionally uses conservative heuristics.  If a field cannot be
    identified confidently, it is omitted rather than guessed.
    """
    metadata: dict[str, str] = {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return metadata

    # First Markdown heading is usually the title in Marker output.
    for line in lines[:15]:
        if line.startswith("# "):
            title = re.sub(r"^#\s+", "", line).strip()
            if title:
                metadata["title"] = title
            break

    # Keep source as a stable fallback identifier.
    metadata["source_id"] = source

    # Conservative year detection from the filename/source.
    year_match = re.search(r"\b(19|20)\d{2}\b", source)
    if year_match:
        metadata["year"] = year_match.group(0)

    return metadata
