"""
Governed Generation Pipeline — Full 4-Phase End-to-End

PROPOSE (MDLM) → DECIDE (G₁-G₇) → PROMOTE (witness commitment) → EXECUTE (output)

This is the GGP working as designed. Four phases, architecturally separated.
PROPOSE ≠ DECIDE ≠ PROMOTE ≠ EXECUTE enforced by module boundaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pipeline.types import Op, Witness, FrameExample, WitnessBundle, WitnessAttestation
from pipeline.mdlm.tokenizer import (
    decode, TOKEN_NAMES, OP_OFFSET, WIT_OFFSET, ATTESTED, WITHHELD,
    G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE, BOS, EOS, PAD, MASK,
)
from pipeline.stages.s4_validate import (
    validate_and_score, TigStatus, Verdict, AdmissibilityResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: PROPOSE — MDLM crystallizes candidate governed structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Candidate:
    """A candidate governed structure crystallized by PROPOSE."""
    tokens: list[int]
    decoded: str
    proposed_at: str = ""


def propose(model, num_candidates: int = 10, seq_len: int = 40,
            g_slots: int = 3, s_slots: int = 4, f_slots: int = 3) -> list[Candidate]:
    """PROPOSE phase: MDLM crystallizes candidate governed structures from noise."""
    from pipeline.mdlm.model import MaskingSchedule, generate

    samples = generate(model, num_candidates, seq_len, MaskingSchedule.HIERARCHICAL, 20,
                       g_slots=g_slots, s_slots=s_slots, f_slots=f_slots)

    candidates = []
    for i in range(num_candidates):
        tokens = samples[i].tolist()
        candidates.append(Candidate(
            tokens=tokens,
            decoded=decode(tokens),
            proposed_at=datetime.now(timezone.utc).isoformat(),
        ))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: DECIDE — G₁-G₇ admissibility gates evaluate candidates
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Decision:
    """Gate decision on a candidate."""
    candidate_index: int
    tig_status: str  # T, U, F
    verdict: str
    rejected_at: Optional[str] = None
    viki_patterns: list[str] = field(default_factory=list)
    decided_at: str = ""


def tokens_to_example(tokens: list[int]) -> Optional[FrameExample]:
    """Convert generated token sequence back to FrameExample for validation."""
    from pipeline.types import (
        ModalityGrounding, OperatorSequence, OperatorExpression,
        SourceProvenance, Tier,
    )

    g_ops, s_ops, f_ops = [], [], []
    witnesses_raw = {}
    current_mod = None

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == G_OPEN: current_mod = "G"
        elif t == S_OPEN: current_mod = "S"
        elif t == F_OPEN: current_mod = "F"
        elif t in (G_CLOSE, S_CLOSE, F_CLOSE): current_mod = None
        elif OP_OFFSET <= t < OP_OFFSET + 15 and current_mod:
            op_name = TOKEN_NAMES[t]
            target = {"G": g_ops, "S": s_ops, "F": f_ops}.get(current_mod)
            if target is not None:
                target.append({"operator": op_name, "evidence": f"generated({op_name})"})
        elif WIT_OFFSET <= t < WIT_OFFSET + 7:
            wit_name = TOKEN_NAMES[t]
            if i + 1 < len(tokens):
                witnesses_raw[wit_name] = {"attested": tokens[i + 1] == ATTESTED, "evidence": f"generated({wit_name})"}
                i += 1
        i += 1

    if not g_ops or not s_ops or not f_ops:
        return None

    example = FrameExample(
        provenance=SourceProvenance(
            source_id="mdlm:generated", tier=Tier.T1, url="mdlm",
            commit_or_version="variant_A", license="generated",
            acquired_at="2026-03-28", artifact_sha256="generated"),
        channel_a=ModalityGrounding(modality="G", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.from_name(op["operator"]), evidence=op["evidence"])
                        for op in g_ops if Op.from_name(op["operator"]) is not None])),
        channel_b=ModalityGrounding(modality="S", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.from_name(op["operator"]), evidence=op["evidence"])
                        for op in s_ops if Op.from_name(op["operator"]) is not None])),
        channel_c=ModalityGrounding(modality="F", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.from_name(op["operator"]), evidence=op["evidence"])
                        for op in f_ops if Op.from_name(op["operator"]) is not None])),
        witnesses=WitnessBundle(),
    )
    for w in Witness:
        wd = witnesses_raw.get(w.canonical_name, {})
        example.witnesses.attestations[w] = WitnessAttestation(
            witness=w, attested=wd.get("attested", False), evidence=wd.get("evidence", ""))
    example.content_hash = example.compute_hash()
    return example


def decide(candidates: list[Candidate]) -> list[tuple[Candidate, Decision, Optional[FrameExample]]]:
    """DECIDE phase: Run each candidate through G₁-G₇ admissibility gates."""

    results = []
    for i, candidate in enumerate(candidates):
        example = tokens_to_example(candidate.tokens)

        if example is None:
            decision = Decision(
                candidate_index=i,
                tig_status="F",
                verdict="FAIL",
                rejected_at="PARSE",
                decided_at=datetime.now(timezone.utc).isoformat(),
            )
            results.append((candidate, decision, None))
            continue

        admissibility = validate_and_score(example)

        decision = Decision(
            candidate_index=i,
            tig_status=admissibility.tig_status.value,
            verdict=admissibility.verdict.value,
            rejected_at=admissibility.rejected_at,
            viki_patterns=[vp.pattern_type for vp in admissibility.viki_patterns],
            decided_at=datetime.now(timezone.utc).isoformat(),
        )
        results.append((candidate, decision, example if admissibility.tig_status == TigStatus.TRUE else None))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: PROMOTE — Witness commitment (irrevocable)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WitnessCommitment:
    """Irrevocable witness commitment for a promoted governed structure."""
    content_hash: str
    witness_bundle_hash: str
    witnesses: dict  # witness_name → {attested, evidence}
    promoted_at: str = ""
    committed: bool = False


def promote(admitted: list[tuple[Candidate, Decision, FrameExample]]) -> list[tuple[FrameExample, WitnessCommitment]]:
    """PROMOTE phase: 7-witness attestation with cryptographic commitment.

    Only T-status candidates from DECIDE enter PROMOTE.
    Unanimity required: all 7 witnesses must attest. Any withholding blocks promotion.
    """
    promoted = []

    for candidate, decision, example in admitted:
        if example is None or decision.tig_status != "T":
            continue

        # Verify witness unanimity
        if not example.witnesses.is_unanimous():
            continue  # Block promotion — witness withholding

        # Build commitment
        bundle_data = json.dumps({
            w.canonical_name: {
                "attested": a.attested,
                "evidence": a.evidence,
            }
            for w, a in example.witnesses.attestations.items()
        }, sort_keys=True)
        bundle_hash = sha256(bundle_data.encode()).hexdigest()

        commitment = WitnessCommitment(
            content_hash=example.content_hash,
            witness_bundle_hash=bundle_hash,
            witnesses={
                w.canonical_name: {"attested": a.attested, "evidence": a.evidence}
                for w, a in example.witnesses.attestations.items()
            },
            promoted_at=datetime.now(timezone.utc).isoformat(),
            committed=True,
        )
        promoted.append((example, commitment))

    return promoted


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: EXECUTE — Output within committed envelope
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GovernedOutput:
    """Final governed output — channel_bly witnessed, committed, traceable."""
    content_hash: str
    gov_structure: dict  # G, S, F operator compositions
    witness_commitment: dict
    provenance: dict
    generated_at: str = ""


def execute(promoted: list[tuple[FrameExample, WitnessCommitment]]) -> list[GovernedOutput]:
    """EXECUTE phase: Produce governed output within committed validity envelope.

    Each output carries its full governance trace:
    - governed structure (what was crystallized)
    - Witness commitment (who attested)
    - Provenance (where it came from)
    """
    outputs = []

    for example, commitment in promoted:
        output = GovernedOutput(
            content_hash=commitment.content_hash,
            gov_structure={
                "G": [{"operator": e.operator.canonical_name, "evidence": e.evidence}
                      for e in example.channel_a.operators.expressions],
                "S": [{"operator": e.operator.canonical_name, "evidence": e.evidence}
                      for e in example.channel_b.operators.expressions],
                "F": [{"operator": e.operator.canonical_name, "evidence": e.evidence}
                      for e in example.channel_c.operators.expressions],
            },
            witness_commitment=asdict(commitment),
            provenance=asdict(example.provenance),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        outputs.append(output)

    return outputs


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineReport:
    """Complete report from one pipeline run."""
    proposed: int
    decided_t: int
    decided_u: int
    decided_f: int
    promoted: int
    executed: int
    viki_detections: int
    elapsed_seconds: float
    outputs: list[dict]


def run_governed_pipeline(model, num_candidates: int = 100,
                          g_slots: int = 3, s_slots: int = 4, f_slots: int = 3) -> PipelineReport:
    """Run the full 4-phase governed generation pipeline.

    PROPOSE → DECIDE → PROMOTE → EXECUTE
    """
    start = time.time()

    # Phase 1: PROPOSE
    candidates = propose(model, num_candidates, g_slots=g_slots, s_slots=s_slots, f_slots=f_slots)

    # Phase 2: DECIDE
    decided = decide(candidates)
    t_count = sum(1 for _, d, _ in decided if d.tig_status == "T")
    u_count = sum(1 for _, d, _ in decided if d.tig_status == "U")
    f_count = sum(1 for _, d, _ in decided if d.tig_status == "F")
    viki_count = sum(len(d.viki_patterns) for _, d, _ in decided)

    admitted = [(c, d, e) for c, d, e in decided if d.tig_status == "T" and e is not None]

    # Phase 3: PROMOTE
    promoted = promote(admitted)

    # Phase 4: EXECUTE
    outputs = execute(promoted)

    elapsed = time.time() - start

    return PipelineReport(
        proposed=num_candidates,
        decided_t=t_count,
        decided_u=u_count,
        decided_f=f_count,
        promoted=len(promoted),
        executed=len(outputs),
        viki_detections=viki_count,
        elapsed_seconds=elapsed,
        outputs=[asdict(o) for o in outputs],
    )
