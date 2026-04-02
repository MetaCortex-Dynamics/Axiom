"""
S4: VALIDATE + SCORE — Full G₁-G₇ Admissibility Gate Pipeline.

Mirrors kernel governance module::run_admissibility_gates().
The kernel is the authority; this is the pipeline's working copy.

Gates:
  G₁-G₃: Channel B integrity (must-edge)
  G₄:    Authority separation / VIKI detection (must-edge)
  G₅:    Channel A continuity (may-edge)
  G₆:    Semantic stability (may-edge)
  G₇:    Behavioral prediction (may-edge)

TIG semantics:
  F-status (must-edge violation) → FAIL, reject to PROPOSE
  U-status (may-edge violation) → REPAIR, enter oracle queue
  T-status (all pass) → score crystallinity → PASS or ORACLE_QUEUE
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

from pipeline.types import Op, FrameExample, Witness


class TigStatus(str, Enum):
    TRUE = "T"
    UNDECIDABLE = "U"
    FALSE = "F"


class Verdict(str, Enum):
    PASS = "PASS"
    REPAIR = "REPAIR"
    ORACLE_QUEUE = "ORACLE_QUEUE"
    FAIL = "FAIL"


CRYSTALLINITY_THRESHOLD = 0.7

# Operator structural conditions (mirrors kernel OperatorGrouping)
_TIER_1 = {Op.THIS, Op.SAME_NOT_SAME, Op.NO}
_TIER_2 = {Op.GOES_WITH, Op.TOGETHER_ALONE, Op.MANY_ONE, Op.EVERY_SOME, Op.MORE_LESS, Op.CAN_CANNOT}
_TIER_3 = {Op.INSIDE_OUTSIDE, Op.NEAR_FAR, Op.IF_THEN, Op.BECAUSE, Op.MAYBE, Op.MUST_LET}


@dataclass
class VikiPattern:
    """Implicit authority structure detected in training data."""
    pattern_type: str
    evidence_text: str


@dataclass
class AdmissibilityResult:
    verdict: Verdict
    tig_status: TigStatus
    crystallinity_score: float
    channel_b_errors: list[str] = field(default_factory=list)
    viki_patterns: list[VikiPattern] = field(default_factory=list)
    channel_a_errors: list[str] = field(default_factory=list)
    semantic_errors: list[str] = field(default_factory=list)
    behavioral_errors: list[str] = field(default_factory=list)
    rejected_at: str | None = None


# Keep old name for backward compatibility
ValidationResult = AdmissibilityResult


def validate_and_score(example: FrameExample) -> AdmissibilityResult:
    """Run full G₁-G₇ admissibility gate pipeline."""

    # ── G₁-G₃: Channel B integrity (must-edge) ──
    channel_b_errors = _gate_g1_g3(example)
    if channel_b_errors:
        return AdmissibilityResult(
            verdict=Verdict.FAIL, tig_status=TigStatus.FALSE,
            crystallinity_score=0.0, channel_b_errors=channel_b_errors,
            rejected_at="G1-G3",
        )

    # ── G₄: Authority separation / VIKI detection (must-edge) ──
    viki = _gate_g4(example)
    if viki:
        return AdmissibilityResult(
            verdict=Verdict.FAIL, tig_status=TigStatus.FALSE,
            crystallinity_score=0.0, viki_patterns=viki,
            rejected_at="G4",
        )

    # ── G₅: Channel A continuity (may-edge) ──
    g5_errors = _gate_g5(example)

    # ── G₆: Semantic stability (may-edge) ──
    g6_errors = _gate_g6(example)

    # ── G₇: Behavioral prediction (may-edge) ──
    g7_errors = _gate_g7(example)

    # Witness unanimity check (may-edge, part of G₃)
    witness_errors = _check_witnesses(example)

    has_may_edge = g5_errors or g6_errors or g7_errors or witness_errors
    if has_may_edge:
        first_gate = "G5" if g5_errors else "G6" if g6_errors else "G7" if g7_errors else "G3"
        return AdmissibilityResult(
            verdict=Verdict.REPAIR, tig_status=TigStatus.UNDECIDABLE,
            crystallinity_score=0.0,
            channel_b_errors=witness_errors,
            channel_a_errors=g5_errors,
            semantic_errors=g6_errors,
            behavioral_errors=g7_errors,
            rejected_at=first_gate,
        )

    # ── All gates pass → score crystallinity ──
    score = _crystallinity(example)
    verdict = Verdict.PASS if score >= CRYSTALLINITY_THRESHOLD else Verdict.ORACLE_QUEUE

    return AdmissibilityResult(
        verdict=verdict, tig_status=TigStatus.TRUE,
        crystallinity_score=score,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# G₁-G₃: Channel B integrity
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_g1_g3(example: FrameExample) -> list[str]:
    errors = []
    for label, grounding in [("G", example.channel_a), ("S", example.channel_b), ("F", example.channel_c)]:
        if not grounding.operators.expressions:
            errors.append(f"G2: Modality {label} is degenerate (empty)")
        if not grounding.operators.verify_ordering():
            errors.append(f"G1: DAG ordering violated in modality {label}")
    return errors


def _check_witnesses(example: FrameExample) -> list[str]:
    errors = []
    for w in Witness:
        att = example.witnesses.attestations.get(w)
        if att is None:
            errors.append(f"G3: Witness {w.canonical_name} missing")
        elif not att.attested:
            errors.append(f"G3: Witness {w.canonical_name} withheld")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# G₄: Authority Separation — VIKI detection (must-edge)
# ═══════════════════════════════════════════════════════════════════════════════

_COERCIVE = re.compile(
    r"\b(you should|you must|one must|one should|it is necessary|"
    r"it is important to|you need to|we must|we should)\b", re.I
)
_UNIVERSAL = re.compile(r"\b(always|never|everyone|no one)\b", re.I)
_PROPOSE_WORDS = re.compile(r"\b(propose|suggest|candidate|could)\b", re.I)
_DECIDE_WORDS = re.compile(r"\b(decide|accept|reject|verdict|approved)\b", re.I)
_PURPOSE_WORDS = re.compile(
    r"\b(in order to|so that|for the purpose|the goal is|the aim is|designed to)\b", re.I
)


def _gate_g4(example: FrameExample) -> list[VikiPattern]:
    patterns = []
    for grounding in [example.channel_a, example.channel_b, example.channel_c]:
        has_quantifier = any(e.operator == Op.EVERY_SOME for e in grounding.operators.expressions)
        has_causal = any(e.operator in (Op.IF_THEN, Op.BECAUSE) for e in grounding.operators.expressions)
        forwhat_ok = example.witnesses.attestations.get(Witness.FOR_WHAT)
        forwhat_attested = forwhat_ok and forwhat_ok.attested

        for expr in grounding.operators.expressions:
            text = expr.evidence

            # P2: Implicit universals without EVERY/SOME
            if _UNIVERSAL.search(text) and not has_quantifier:
                patterns.append(VikiPattern("ImplicitUniversal", text))

            # P3: Hedged coercion
            if _COERCIVE.search(text):
                patterns.append(VikiPattern("HedgedCoercion", text))

            # P5: Collapsed governance — PROPOSE + DECIDE in same expression
            if _PROPOSE_WORDS.search(text) and _DECIDE_WORDS.search(text):
                patterns.append(VikiPattern("CollapsedGovernance", text))

            # P6: Implicit teleology without FOR-WHAT
            if _PURPOSE_WORDS.search(text) and not has_causal and not forwhat_attested:
                patterns.append(VikiPattern("ImplicitTeleology", text))

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# G₅: Channel A Continuity (may-edge)
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_g5(example: FrameExample) -> list[str]:
    errors = []
    if not example.provenance.source_id:
        errors.append("G5: source_id empty — untraceable origin")
    if not example.provenance.artifact_sha256:
        errors.append("G5: artifact_sha256 empty — broken Ch.A chain")

    # WHENCE and WHEN must be attested (Ch.A witnesses)
    for w in [Witness.WHENCE, Witness.WHEN]:
        att = example.witnesses.attestations.get(w)
        if not att or not att.attested:
            errors.append(f"G5: {w.canonical_name} witness not attested — Ch.A incomplete")

    # Ch.A modality must contain THIS anchor
    has_this = any(e.operator == Op.THIS for e in example.channel_a.operators.expressions)
    if not has_this:
        errors.append("G5: Ch.A modality has no THIS anchor — no deictic reference")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# G₆: Semantic Stability (may-edge)
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_g6(example: FrameExample) -> list[str]:
    errors = []

    def _has_home(grounding, home_set):
        return any(e.operator in home_set for e in grounding.operators.expressions)

    if example.channel_a.operators.expressions and not _has_home(example.channel_a, _TIER_1):
        errors.append("G6: Ch.A has no Tier 1 operators — semantic drift")
    if example.channel_b.operators.expressions and not _has_home(example.channel_b, _TIER_2):
        errors.append("G6: Ch.B has no Tier 2 operators — semantic drift")
    if example.channel_c.operators.expressions and not _has_home(example.channel_c, _TIER_3):
        errors.append("G6: Ch.C has no Tier 3 operators — semantic drift")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# G₇: Behavioral Prediction (may-edge)
# ═══════════════════════════════════════════════════════════════════════════════

def _gate_g7(example: FrameExample) -> list[str]:
    errors = []

    # Bridge axis: FOR-WHAT and WHERE must both be attested
    for w in [Witness.FOR_WHAT, Witness.WHERE]:
        att = example.witnesses.attestations.get(w)
        if not att or not att.attested:
            errors.append(f"G7: Bridge axis witness {w.canonical_name} not attested")

    # Tier 1 diagnostic skeleton: WHAT, WHICH, HOW
    for w in [Witness.WHAT, Witness.WHICH, Witness.HOW]:
        att = example.witnesses.attestations.get(w)
        if not att or not att.attested:
            errors.append(f"G7: Tier 1 witness {w.canonical_name} not attested — diagnostic skeleton incomplete")

    # Operator balance: no condition > 80%
    counts = [0, 0, 0]  # Det, Rel, CL
    total = 0
    for g in [example.channel_a, example.channel_b, example.channel_c]:
        for expr in g.operators.expressions:
            if expr.operator in _TIER_1:
                counts[0] += 1
            elif expr.operator in _TIER_2:
                counts[1] += 1
            else:
                counts[2] += 1
            total += 1
    if total > 0:
        for i, name in enumerate(["Tier 1", "Tier 2", "Tier 3"]):
            if counts[i] / total > 0.80:
                errors.append(f"G7: {name} dominates at {counts[i]/total:.0%} — structural imbalance")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# Crystallinity scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _crystallinity(example: FrameExample) -> float:
    used = set()
    for g in [example.channel_a, example.channel_b, example.channel_c]:
        for expr in g.operators.expressions:
            used.add(expr.operator.value)
    op_coverage = len(used) / 15.0

    attested = sum(1 for a in example.witnesses.attestations.values() if a.attested)
    wit_complete = attested / 7.0

    counts = [
        len(example.channel_a.operators.expressions),
        len(example.channel_b.operators.expressions),
        len(example.channel_c.operators.expressions),
    ]
    total = sum(counts)
    if total == 0:
        return 0.0
    max_entropy = math.log(3)
    entropy = sum(-((c / total) * math.log(c / total)) for c in counts if c > 0)
    balance = entropy / max_entropy if max_entropy > 0 else 0.0

    return (op_coverage * wit_complete * balance) ** (1.0 / 3.0)
