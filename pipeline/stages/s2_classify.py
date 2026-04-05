"""
S2: CLASSIFY — Technical vs non-technical classification.

Per specification:
  Rule-based classifier first (keyword density, channel_b markers).
  High-confidence → S3. Ambiguous → oracle queue.

Invariant: Every segment is classified. No segment is unclassified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pipeline.stages.s1_segment import Segment
from pipeline.types import ExclusionReason


class Classification(str, Enum):
    TECHNICAL = "TECHNICAL"
    NON_TECHNICAL = "NON_TECHNICAL"
    AMBIGUOUS = "AMBIGUOUS"  # → oracle queue


@dataclass
class ClassifiedSegment:
    segment: Segment
    classification: Classification
    confidence: float  # 0.0 - 1.0
    exclusion_reason: Optional[ExclusionReason] = None  # if NON_TECHNICAL


# Keyword sets for rule-based classification
_TECHNICAL_KEYWORDS = {
    "module", "endmodule", "input", "output", "wire", "reg", "assign",
    "always", "posedge", "negedge", "assert", "assume", "cover",
    "property", "constraint", "parameter", "localparam", "generate",
    "function", "task", "begin", "end", "if", "else", "case",
    "invariant", "specification", "requirement", "shall", "must",
    "verification", "equivalence", "formal", "synthesis", "netlist",
}

_NON_TECHNICAL_PATTERNS = [
    (re.compile(r"(?i)copyright|license|all rights reserved|permission is hereby"),
     ExclusionReason.LICENSE_HEADER),
    (re.compile(r"(?i)changelog|release notes|version history|what.s new"),
     ExclusionReason.CHANGELOG),
    (re.compile(r"(?i)table of contents|page \d+|index$"),
     ExclusionReason.FORMATTING),
    (re.compile(r"(?i)acknowledgments?|thanks to|we would like to"),
     ExclusionReason.EDITORIAL),
    (re.compile(r"(?i)disclaimer|no warranty|as.is"),
     ExclusionReason.BOILERPLATE),
    # Platform injection — ChatGPT filecite metadata, not user content
    (re.compile(r"fileciteturn\d+file\d+"),
     ExclusionReason.BOILERPLATE),
]


def classify(seg: Segment) -> ClassifiedSegment:
    """Classify a segment as TECHNICAL, NON_TECHNICAL, or AMBIGUOUS."""
    text = seg.text

    # Whitespace-only check
    if not text.strip():
        return ClassifiedSegment(
            segment=seg,
            classification=Classification.NON_TECHNICAL,
            confidence=1.0,
            exclusion_reason=ExclusionReason.WHITESPACE,
        )

    # Non-technical pattern matching
    for pattern, reason in _NON_TECHNICAL_PATTERNS:
        if pattern.search(text):
            return ClassifiedSegment(
                segment=seg,
                classification=Classification.NON_TECHNICAL,
                confidence=0.9,
                exclusion_reason=reason,
            )

    # Technical keyword density
    words = set(re.findall(r"\w+", text.lower()))
    tech_count = len(words & _TECHNICAL_KEYWORDS)
    total_words = max(len(words), 1)
    tech_density = tech_count / total_words

    # Channel B markers
    has_verilog = bool(re.search(r"\b(module|endmodule|always|assign)\b", text))
    has_assertion = bool(re.search(r"\b(assert|assume|cover)\s+property", text))
    has_formal = bool(re.search(r"\b(invariant|specification|requirement)\b", text, re.I))

    if has_verilog or has_assertion:
        return ClassifiedSegment(
            segment=seg,
            classification=Classification.TECHNICAL,
            confidence=0.95,
        )

    if tech_density > 0.15 or has_formal:
        return ClassifiedSegment(
            segment=seg,
            classification=Classification.TECHNICAL,
            confidence=0.7 + tech_density,
        )

    if tech_density < 0.05:
        return ClassifiedSegment(
            segment=seg,
            classification=Classification.NON_TECHNICAL,
            confidence=0.7,
            exclusion_reason=ExclusionReason.BOILERPLATE,
        )

    # Ambiguous — send to oracle
    return ClassifiedSegment(
        segment=seg,
        classification=Classification.AMBIGUOUS,
        confidence=tech_density,
    )
