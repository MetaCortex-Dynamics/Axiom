"""
Core types for the Governed Decomposition Pipeline.

These mirror the kernel's Rust types (kernel governance types)
as Python dataclasses for pipeline processing. The kernel is the authority;
these are the pipeline's working copies.

Interface contract with kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from hashlib import sha256
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 15 GROUNDING OPERATORS — mirrors kernel GroundingOperator
# ═══════════════════════════════════════════════════════════════════════════════

class Op(IntEnum):
    """15 grounding operators in admissibility DAG order."""
    THIS           = 0
    GOES_WITH      = 1
    MANY_ONE       = 2
    EVERY_SOME     = 3
    NO             = 4
    IF_THEN        = 5
    BECAUSE        = 6
    SAME_NOT_SAME  = 7
    INSIDE_OUTSIDE = 8
    CAN_CANNOT     = 9
    MAYBE          = 10
    MUST_LET       = 11
    TOGETHER_ALONE = 12
    MORE_LESS      = 13
    NEAR_FAR       = 14

    @property
    def canonical_name(self) -> str:
        return _OP_NAMES[self.value]

    @classmethod
    def from_name(cls, name: str) -> Optional[Op]:
        return _OP_BY_NAME.get(name)


_OP_NAMES = [
    "THIS", "GOES-WITH", "MANY/ONE", "EVERY/SOME", "NO",
    "IF/THEN", "BECAUSE", "SAME/NOT-SAME", "INSIDE/OUTSIDE",
    "CAN/CANNOT", "MAYBE", "MUST/LET", "TOGETHER/ALONE",
    "MORE/LESS", "NEAR/FAR",
]
_OP_BY_NAME = {name: Op(i) for i, name in enumerate(_OP_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════════
# 7 THEOREM-DERIVED WITNESSES — mirrors kernel InterrogativeWitness
# ═══════════════════════════════════════════════════════════════════════════════

class Witness(IntEnum):
    """7 interrogative witnesses (theorem-derived, NOT folk WHO/WHY)."""
    WHAT     = 0
    WHERE    = 1
    WHICH    = 2
    WHEN     = 3
    FOR_WHAT = 4
    HOW      = 5
    WHENCE   = 6

    @property
    def canonical_name(self) -> str:
        return _WIT_NAMES[self.value]

    @property
    def modality(self) -> str:
        """modality assignment: G(WHENCE,WHEN), S(WHAT,WHERE,WHICH), F(FOR-WHAT,HOW)."""
        return _WIT_MODALITY[self.value]

    @property
    def is_bridge_axis(self) -> bool:
        """FOR-WHAT x WHERE = critical axis."""
        return self in (Witness.FOR_WHAT, Witness.WHERE)

    @property
    def fragility_tier(self) -> int:
        """Tier 1 (descriptive: WHAT,WHICH,HOW) persists after normative collapse. Tier 2 = rest."""
        return 1 if self in (Witness.WHAT, Witness.WHICH, Witness.HOW) else 2


_WIT_NAMES = ["WHAT", "WHERE", "WHICH", "WHEN", "FOR-WHAT", "HOW", "WHENCE"]
_WIT_MODALITY = ["S", "S", "S", "G", "F", "F", "G"]


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS TIERS
# ═══════════════════════════════════════════════════════════════════════════════

class Tier(str, Enum):
    T1 = "T1"  # EDA tool output
    T2 = "T2"  # RTL design documents
    T3 = "T3"  # Formal property specifications


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE — Provenance record format
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceProvenance:
    source_id: str
    tier: Tier
    url: str
    commit_or_version: str
    license: str
    acquired_at: str  # ISO 8601
    artifact_sha256: str
    acquired_by: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATOR EXPRESSIONS + governed EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OperatorExpression:
    operator: Op
    arguments: list[str] = field(default_factory=list)
    evidence: str = ""


@dataclass
class OperatorSequence:
    expressions: list[OperatorExpression] = field(default_factory=list)

    def verify_ordering(self) -> bool:
        for i in range(1, len(self.expressions)):
            if self.expressions[i].operator < self.expressions[i - 1].operator:
                return False
        return True


@dataclass
class ModalityGrounding:
    """A single governed modality's operator sequence."""
    modality: str  # "G", "S", "F"
    operators: OperatorSequence = field(default_factory=OperatorSequence)


@dataclass
class WitnessAttestation:
    witness: Witness
    attested: bool
    evidence: str = ""
    reason: str = ""  # if withheld


@dataclass
class WitnessBundle:
    attestations: dict[Witness, WitnessAttestation] = field(default_factory=dict)

    def is_unanimous(self) -> bool:
        if len(self.attestations) != len(Witness):
            return False
        return all(a.attested for a in self.attestations.values())


@dataclass
class FrameExample:
    provenance: SourceProvenance
    channel_a: ModalityGrounding
    channel_b: ModalityGrounding
    channel_c: ModalityGrounding
    witnesses: WitnessBundle
    content_hash: str = ""

    def compute_hash(self) -> str:
        h = sha256()
        h.update(self.provenance.source_id.encode())
        for g in [self.channel_a, self.channel_b, self.channel_c]:
            for expr in g.operators.expressions:
                h.update(bytes([expr.operator.value]))
                h.update(expr.evidence.encode())
        return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# EXCLUSION MANIFEST — Exclusion manifest
# ═══════════════════════════════════════════════════════════════════════════════

class ExclusionReason(str, Enum):
    LICENSE_HEADER     = "LicenseHeader"
    CHANGELOG          = "Changelog"
    FORMATTING         = "FormattingArtifact"
    EDITORIAL          = "EditorialNote"
    BOILERPLATE        = "Boilerplate"
    WHITESPACE         = "WhitespaceOnly"
    DUPLICATE          = "DuplicateContent"
    UNTRANSLATABLE     = "Untranslatable"


class Classifier(str, Enum):
    RULE   = "RULE"
    ORACLE = "ORACLE"


@dataclass
class ExclusionRecord:
    source_id: str
    byte_range: tuple[int, int]
    segment_text: str
    reason: ExclusionReason
    classified_by: Classifier
    classified_at: str = ""  # ISO 8601


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE — Oracle protocol
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStage(str, Enum):
    S2_CLASSIFY = "S2_CLASSIFY"
    S4_VALIDATE = "S4_VALIDATE"


class OracleVerdict(str, Enum):
    TECHNICAL      = "TECHNICAL"
    NON_TECHNICAL  = "NON_TECHNICAL"
    ACCEPT_AS_IS   = "ACCEPT_AS_IS"
    REVISE         = "REVISE"
    REJECT         = "REJECT"


@dataclass
class OracleReviewItem:
    item_id: str
    source: SourceProvenance
    segment_text: str
    pipeline_stage: PipelineStage
    automated_verdict: str
    escalation_reason: str


@dataclass
class OracleDecision:
    item_id: str
    decision: OracleVerdict
    revised_example: Optional[FrameExample] = None
    exclusion_reason: Optional[ExclusionReason] = None
    notes: str = ""
    decided_at: str = ""
    decided_by: str = ""
