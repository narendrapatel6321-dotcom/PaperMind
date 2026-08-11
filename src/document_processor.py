"""Document ingestion and structure-aware chunking for PaperMind.

The input corpus consists of research papers extracted to TXT with Marker.
This module cleans Marker artifacts, detects paper structure, preserves useful
scientific blocks (tables, captions and equations), and creates retrieval
chunks with source/section/page metadata.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    text: str
    source: str
    path: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Section:
    title: str
    text: str
    number: str | None = None
    level: int = 1
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ContentBlock:
    text: str
    content_type: str = "text"
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TextFileLoader:
    """Load all non-empty TXT research papers from a directory."""

    def load_documents(self, data_dir: Path) -> list[RawDocument]:
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

            documents.append(
                RawDocument(
                    text=text,
                    source=path.stem,
                    path=str(path),
                    metadata=_extract_paper_metadata(text, path.stem),
                )
            )

        logger.info("Loaded %d document(s) from %s", len(documents), data_dir)
        return documents

    def load(self, data_dir: Path) -> list[tuple[str, str]]:
        """Backward-compatible interface used by the original project."""
        return [(d.text, d.source) for d in self.load_documents(data_dir)]


# ---------------------------------------------------------------------------
# Marker cleaning
# ---------------------------------------------------------------------------

class MarkerCleaner:
    """Remove Marker-only syntax while preserving Markdown/scientific text."""

    PAGE_RE = re.compile(
        r"<span\s+id=[\"']page-(\d+)-\d+[\"']\s*>\s*</span>",
        re.IGNORECASE,
    )
    IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
    HTML_TAG_RE = re.compile(
        r"</?(?!sup\b|sub\b|br\b)[a-zA-Z][^>]*>", re.IGNORECASE
    )
    PAGE_ANCHOR_RE = re.compile(r"\(#page-\d+-\d+\)")

    def clean(self, text: str) -> str:
        def page_repl(match: re.Match[str]) -> str:
            return f"\n<!-- PAGE:{match.group(1)} -->\n"

        text = self.PAGE_RE.sub(page_repl, text)
        text = self.IMAGE_RE.sub("", text)
        text = self.LINK_RE.sub(r"\1", text)
        text = self.PAGE_ANCHOR_RE.sub("", text)
        text = html.unescape(text)
        text = self.HTML_TAG_RE.sub("", text)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

        # Common Marker/mailto artifacts.
        text = re.sub(r"\[([^\]]+)\]\(mailto\\?:[^)]*\)", r"\1", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

class SectionParser:
    """Detect Markdown, numbered, uppercase and simple bold headings."""

    MD_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")
    NUMBERED_RE = re.compile(
        r"^(\d+(?:\.\d+)*)(?:[.)])?\s+([A-Za-z][^\n]{1,120})$"
    )
    UPPER_RE = re.compile(r"^[A-Z][A-Z0-9\s:,&'()/+\-]{2,100}$")
    BOLD_RE = re.compile(r"^\*\*([^*]{2,120})\*\*$")

    def parse(self, text: str) -> list[Section]:
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

        for raw in text.splitlines():
            line = raw.strip()
            page = re.fullmatch(r"<!--\s*PAGE:(\d+)\s*-->", line)
            if page:
                page_no = int(page.group(1))
                if current_page_start is None:
                    current_page_start = page_no
                current_page_end = page_no
                # Keep the marker as a boundary for block-level page tracking.
                current_lines.append(line)
                continue

            heading = self._parse_heading(line)
            if heading is not None:
                flush()
                title, number, level = heading
                current_title = title
                current_number = number
                current_level = level
                current_page_start = current_page_end
                continue

            current_lines.append(raw)

        flush()
        return sections

    def _parse_heading(self, line: str) -> tuple[str, str | None, int] | None:
        if not line:
            return None

        md = self.MD_RE.match(line)
        if md:
            title = md.group(2).strip()
            if not title:
                return None
            number, title = _split_section_number(title)
            return title, number, len(md.group(1))

        numbered = self.NUMBERED_RE.match(line)
        if numbered and self._reasonable_heading(numbered.group(2)):
            return numbered.group(2).strip(), numbered.group(1), 2

        bold = self.BOLD_RE.match(line)
        if bold:
            title = bold.group(1).strip()
            # Avoid treating labels such as **Original input:** as sections.
            if not title.endswith(":") and self._reasonable_heading(title):
                number, title = _split_section_number(title)
                return title, number, 2

        if (
            self.UPPER_RE.match(line)
            and len(line) <= 100
            and not line.endswith(".")
        ):
            return line, None, 1

        return None

    @staticmethod
    def _reasonable_heading(text: str) -> bool:
        if not text or len(text) > 120:
            return False
        if text.endswith(".") and len(text.split()) > 8:
            return False
        return True


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

class FrontMatterFilter:
    """Remove author/affiliation/email noise before the abstract."""

    EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
    URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
    AFFILIATION_TERMS = (
        "university", "institute", "laboratory", "department", "nvidia",
        "google", "microsoft", "openai", "facebook", "meta", "ibm",
    )
    
    def filter(self, sections: list[Section], title: str | None = None, ) -> list[Section]:
        result: list[Section] = []
        for section in sections:
            if section.title != "Front Matter":
                result.append(section)
                continue

            kept: list[str] = []
            for paragraph in re.split(r"\n\s*\n", section.text):
                p = paragraph.strip()

                if not p:
                    continue

                if title and " ".join(p.split()) == " ".join(title.split()):
                    continue

                if self._is_noise(p):
                    continue
                kept.append(p)

            if kept:
                result.append(
                    Section(
                        title=section.title,
                        text="\n\n".join(kept),
                        number=section.number,
                        level=section.level,
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                )
        return result

    @staticmethod
    def _is_noise(text: str) -> bool:
        compact = " ".join(text.split())

        if not compact:
            return True

        # Isolated Markdown artifacts.
        if re.fullmatch(r"#{1,6}", compact):
            return True

        # Isolated punctuation / table remnants.
        if re.fullmatch(r"[-–—|*_~•·.]+", compact):
            return True

        # Isolated email.
        if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", compact):
            return True

        # Isolated URL.
        if re.fullmatch(
            r"(?:https?://|www\.)\S+",
            compact,
            re.IGNORECASE,
        ):
            return True

        return False


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

class BlockExtractor:
    """Extract paragraphs, tables, captions and page-aware blocks."""

    TABLE_SEPARATOR_RE = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
    CAPTION_RE = re.compile(
        r"^\s*(?:Figure|Fig\.|Table)\s+\d+[:.]", re.IGNORECASE
    )

    def extract(self, section: Section) -> list[ContentBlock]:
        lines = section.text.splitlines()
        blocks: list[ContentBlock] = []
        current: list[str] = []
        current_page: int | None = section.page_start

        def flush() -> None:
            nonlocal current
            text = "\n".join(current).strip()
            current = []
            if not text:
                return
            if self._is_noise(text):
                return
            blocks.append(
                ContentBlock(
                    text=text,
                    content_type="text",
                    page_start=current_page,
                    page_end=current_page,
                )
            )

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            page = re.fullmatch(r"<!--\s*PAGE:(\d+)\s*-->", line)
            if page:
                flush()
                current_page = int(page.group(1))
                i += 1
                continue

            if (
                i + 1 < len(lines)
                and self.TABLE_ROW_RE.match(line)
                and self.TABLE_SEPARATOR_RE.match(lines[i + 1].strip())
            ):
                flush()
                table_lines = [line, lines[i + 1].strip()]
                i += 2
                while i < len(lines) and self.TABLE_ROW_RE.match(lines[i].strip()):
                    table_lines.append(lines[i].strip())
                    i += 1
                blocks.append(
                    ContentBlock(
                        text="\n".join(table_lines),
                        content_type="table",
                        page_start=current_page,
                        page_end=current_page,
                    )
                )
                continue

            if self.CAPTION_RE.match(line):
                flush()
                blocks.append(
                    ContentBlock(
                        text=line,
                        content_type="caption",
                        page_start=current_page,
                        page_end=current_page,
                    )
                )
                i += 1
                continue

            if not line:
                flush()
                i += 1
                continue

            current.append(lines[i])
            i += 1

        flush()
        return blocks

    @staticmethod
    def _is_noise(text: str) -> bool:
        compact = " ".join(text.split())
        if not compact:
            return True
        if re.fullmatch(r"#{1,6}", compact):
            return True
        if re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", compact):
            return True
        if re.fullmatch(r"(?:https?://|www\.)\S+", compact, re.IGNORECASE):
            return True
        return False


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------
def _normalize_section_title(title: str) -> str:
    """Normalize formatting for section comparisons and metadata."""
    title = re.sub(r"[*_`]+", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


class StructureAwareChunker:
    """Assemble natural content blocks into bounded retrieval chunks."""

    def __init__(
        self,
        target_chars: int = 1500,
        max_chars: int = 2400,
        overlap_chars: int = 200,
        min_chunk_chars: int = 150,
        exclude_references: bool = True,
    ) -> None:
        if target_chars <= 0 or max_chars < target_chars:
            raise ValueError("Require 0 < target_chars <= max_chars")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be >= 0 and < target_chars")
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be >= 0")

        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.exclude_references = exclude_references
    
    def chunk_sections(
        self,
        sections: Iterable[Section],
        source: str,
        base_metadata: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        base_metadata = dict(base_metadata or {})
        chunk_id = 0
        extractor = BlockExtractor()

        for section in sections:
            section_title = _normalize_section_title(section.title)
            normalized_title = section_title.lower()

            if self.exclude_references and normalized_title in {
                "references",
                "bibliography",
                "references and notes",
                }:
                    continue

            if normalized_title in {
                "acknowledgements",
                "acknowledgments",
                "contents",
                }:
                    continue

            blocks = extractor.extract(section)
            assembled = self._assemble(blocks)

            for text, types, page_start, page_end in assembled:
                if not text.strip():
                    continue

                metadata = {
                            **base_metadata,
                            "section": section_title,
                            "section_level": str(section.level),
                            "content_type": self._content_type(types),
                            }
                if section.number:
                    metadata["section_number"] = section.number
                if page_start is not None:
                    metadata["page_start"] = str(page_start)
                if page_end is not None:
                    metadata["page_end"] = str(page_end)

                chunks.append(
                    DocumentChunk(
                        text=text.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        metadata=metadata,
                    )
                )
                chunk_id += 1

        logger.info("%s -> %d chunks", source, len(chunks))
        return chunks

    def _assemble(self, blocks: list[ContentBlock]):
        """Pack blocks until target size; never split a natural block."""
        results = []
        current: list[ContentBlock] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            text = "\n\n".join(b.text.strip() for b in current if b.text.strip())
            if text:
                results.append((
                    text,
                    [b.content_type for b in current],
                    current[0].page_start,
                    current[-1].page_end,
                ))
            current = []
            current_len = 0

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            # Oversized atomic block: preserve as much as possible, then fall
            # back to a word boundary split.
            if len(text) > self.max_chars and block.content_type == "table":
                flush()
                for piece in self._split_long_text(text):
                    results.append((piece, ["table"], block.page_start, block.page_end))
                continue

            added = len(text) + (2 if current else 0)
            if current and current_len + added > self.max_chars:
                flush()

            current.append(block)
            current_len += len(text) + (2 if len(current) > 1 else 0)

            # Stop only at block boundaries. This is the key difference from
            # the old fixed-character chunker.
            if current_len >= self.target_chars:
                flush()

        flush()
        return self._merge_short(results)

    def _merge_short(self, chunks):
        """Merge short chunks with neighbors instead of deleting them."""
        if not chunks:
            return []

        merged = []
        for chunk in chunks:
            text, types, start, end = chunk
            if merged and len(text) < self.min_chunk_chars:
                prev_text, prev_types, prev_start, prev_end = merged[-1]
                if len(prev_text) + len(text) + 2 <= self.max_chars:
                    merged[-1] = (
                        prev_text + "\n\n" + text,
                        prev_types + types,
                        prev_start,
                        end or prev_end,
                    )
                    continue
            merged.append(chunk)

        # Handle a short first chunk by merging forward.
        if len(merged) >= 2 and len(merged[0][0]) < self.min_chunk_chars:
            first, second = merged[0], merged[1]
            if len(first[0]) + len(second[0]) + 2 <= self.max_chars:
                merged = [(
                    first[0] + "\n\n" + second[0],
                    first[1] + second[1],
                    first[2] or second[2],
                    second[3] or first[3],
                )] + merged[2:]

        return merged

    def _split_long_text(self, text: str) -> list[str]:
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            added = len(word) + (1 if current else 0)
            if current and current_len + added > self.max_chars:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            current.append(word)
            current_len += added

        if current:
            pieces.append(" ".join(current))
        return pieces

    @staticmethod
    def _content_type(types: list[str]) -> str:
        unique = list(dict.fromkeys(types))
        return unique[0] if len(unique) == 1 else "mixed"


# ---------------------------------------------------------------------------
# High-level processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Process a directory of Marker TXT papers into retrieval chunks."""

    def __init__(
        self,
        target_chars: int = 1500,
        max_chars: int = 2400,
        overlap_chars: int = 200,
        min_chunk_chars: int = 150,
        exclude_references: bool = True,
    ) -> None:
        self.loader = TextFileLoader()
        self.cleaner = MarkerCleaner()
        self.parser = SectionParser()
        self.front_matter = FrontMatterFilter()
        self.chunker = StructureAwareChunker(
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
            exclude_references=exclude_references,
        )

    def process_directory(self, data_dir: Path) -> list[DocumentChunk]:
        documents = self.loader.load_documents(data_dir)
        if not documents:
            raise FileNotFoundError(
                f"No non-empty .txt documents found in '{data_dir}'."
            )

        all_chunks: list[DocumentChunk] = []
        for document in documents:
            cleaned = self.cleaner.clean(document.text)
            sections = self.parser.parse(cleaned)
            sections = self.front_matter.filter(sections)

            all_chunks.extend(
                self.chunker.chunk_sections(
                    sections,
                    source=document.source,
                    base_metadata={
                        **document.metadata,
                        "source_file": Path(document.path).name,
                    },
                )
            )

        logger.info(
            "Processed %d document(s) -> %d total chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

class DocumentChunker:
    """Compatibility wrapper for the original public API."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self._processor = DocumentProcessor(
            target_chars=chunk_size,
            max_chars=max(chunk_size, 2400),
            overlap_chars=min(chunk_overlap, max(0, chunk_size // 4)),
            min_chunk_chars=150,
        )

    def chunk(self, text: str, source: str) -> list[DocumentChunk]:
        cleaned = self._processor.cleaner.clean(text)
        sections = self._processor.front_matter.filter(
            self._processor.parser.parse(cleaned)
        )
        return self._processor.chunker.chunk_sections(sections, source)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _extract_paper_metadata(text: str, source: str) -> dict[str, str]:
    metadata = {"source_id": source}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:20]:
        if line.startswith("# "):
            title = line[2:].strip()
            if title:
                metadata["title"] = title
            break

    year = re.search(r"\b(19|20)\d{2}\b", source)
    if year:
        metadata["year"] = year.group(0)

    return metadata


def _split_section_number(title: str) -> tuple[str | None, str]:
    match = re.match(
        r"^(\d+(?:\.\d+)*)(?:[.)])?\s+(.+)$", title.strip()
    )
    if match:
        return match.group(1), match.group(2).strip()
    return None, title.strip()
