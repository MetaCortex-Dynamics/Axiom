"""
S1: SEGMENT — Split input into atomic units.

Per SPEC-PIPELINE-001 Part B.2:
  Verilog/SystemVerilog: split by module boundary.
  Natural language docs: split by section/paragraph.
  SVA: split by assertion.
  CSV/JSON: split by record.

Invariant: Union of all segment byte-ranges = source file minus whitespace-only gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from pipeline.types import SourceProvenance, Tier


@dataclass
class Segment:
    """An atomic unit of source content with byte-range traceability."""
    source: SourceProvenance
    byte_range: tuple[int, int]  # (start, end) inclusive
    text: str
    segment_type: str  # "module", "section", "assertion", "record"


def segment(source: SourceProvenance, content: bytes) -> Iterator[Segment]:
    """Split source content into atomic segments based on tier and format."""
    text = content.decode("utf-8", errors="replace")

    if source.tier == Tier.T1:
        yield from _segment_t1(source, text)
    elif source.tier == Tier.T2:
        yield from _segment_t2(source, text)
    elif source.tier == Tier.T3:
        yield from _segment_t3(source, text)


def _segment_t1(source: SourceProvenance, text: str) -> Iterator[Segment]:
    """T1: EDA output — split by record (CSV line or JSON object)."""
    offset = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            start = text.index(line, offset)
            end = start + len(line) - 1
            yield Segment(
                source=source,
                byte_range=(start, end),
                text=stripped,
                segment_type="record",
            )
        offset = text.index(line, offset) + len(line)


def _segment_t2(source: SourceProvenance, text: str) -> Iterator[Segment]:
    """T2: RTL docs — split by module boundary or section heading."""
    # Verilog module boundary detection
    import re
    module_pattern = re.compile(
        r"(module\s+\w+[\s\S]*?endmodule)", re.MULTILINE
    )
    last_end = 0
    for match in module_pattern.finditer(text):
        # Emit any pre-module text as a section
        if match.start() > last_end:
            pre_text = text[last_end:match.start()].strip()
            if pre_text:
                yield Segment(
                    source=source,
                    byte_range=(last_end, match.start() - 1),
                    text=pre_text,
                    segment_type="section",
                )
        yield Segment(
            source=source,
            byte_range=(match.start(), match.end() - 1),
            text=match.group(0),
            segment_type="module",
        )
        last_end = match.end()

    # Trailing content
    if last_end < len(text):
        trailing = text[last_end:].strip()
        if trailing:
            yield Segment(
                source=source,
                byte_range=(last_end, len(text) - 1),
                text=trailing,
                segment_type="section",
            )


def _segment_t3(source: SourceProvenance, text: str) -> Iterator[Segment]:
    """T3: Formal properties — split by assertion."""
    import re
    assertion_pattern = re.compile(
        r"((?:assert|assume|cover)\s+property\s*\([\s\S]*?\)\s*;)", re.MULTILINE
    )
    last_end = 0
    for match in assertion_pattern.finditer(text):
        if match.start() > last_end:
            pre_text = text[last_end:match.start()].strip()
            if pre_text:
                yield Segment(
                    source=source,
                    byte_range=(last_end, match.start() - 1),
                    text=pre_text,
                    segment_type="section",
                )
        yield Segment(
            source=source,
            byte_range=(match.start(), match.end() - 1),
            text=match.group(0),
            segment_type="assertion",
        )
        last_end = match.end()

    if last_end < len(text):
        trailing = text[last_end:].strip()
        if trailing:
            yield Segment(
                source=source,
                byte_range=(last_end, len(text) - 1),
                text=trailing,
                segment_type="section",
            )
