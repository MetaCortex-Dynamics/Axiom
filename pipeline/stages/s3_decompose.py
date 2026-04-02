"""
S3: DECOMPOSE — Prose to (Ch.A, Ch.B, Ch.C) in 15-operator vocabulary.

Per SPEC-PIPELINE-001 Part B.2:
  (a) Identify channel_a content → Ch.A using WHENCE, WHEN
  (b) Identify channel_b content → Ch.B using WHAT, WHERE, WHICH
  (c) Identify channel_c content → Ch.C using FOR-WHAT, HOW

T1 bypass: CecCert sources call extract_from_cert() directly.
This module handles T2 (RTL docs) and T3 (formal properties).

THIS IS THE CORE IP. The decomposer patterns are protected.
"""

from __future__ import annotations

from pipeline.stages.s1_segment import Segment
from pipeline.stages.s2_classify import ClassifiedSegment, Classification
from pipeline.types import (
    ModalityGrounding, Op, OperatorExpression, OperatorSequence,
    SourceProvenance, Tier, FrameExample, Witness, WitnessAttestation,
    WitnessBundle,
)


def decompose(classified: ClassifiedSegment) -> FrameExample | None:
    """Decompose a TECHNICAL segment into a FrameExample.

    Returns None if the segment cannot be decomposed (should not happen
    for properly classified TECHNICAL segments — log as Untranslatable).
    """
    seg = classified.segment
    if classified.classification != Classification.TECHNICAL:
        return None

    if seg.source.tier == Tier.T1:
        return _decompose_t1(seg)
    elif seg.source.tier == Tier.T2:
        return _decompose_t2(seg)
    elif seg.source.tier == Tier.T3:
        # T3 includes chat archives — use content-aware routing
        if seg.segment_type.startswith("chat_"):
            return _decompose_chat(seg)
        return _decompose_t3(seg)
    return None


def _decompose_t1(seg: Segment) -> FrameExample:
    """T1 bypass — CecCert extraction handled by kernel.

    At pipeline level, T1 records are already structured. The kernel's
    extract_from_cert() does the actual projection. Here we wrap
    the segment text as a minimal FrameExample for T1 non-cert records
    (e.g., CSV rows, Yosys JSON fragments, Kernel source).
    """
    text = seg.text
    text_lower = text.lower()

    # Kernel source — extract operator structure from code
    if any(kw in text_lower for kw in ["module ", "pub fn ", "pub mod ", "record ", "trait ", "impl "]):
        return _build_example(
            seg,
            g_ops=[
                (Op.THIS, f"this(source_module) — T1 code unit"),
                (Op.NO, f"no(floating_point) — deterministic arithmetic"),
            ],
            s_ops=[
                (Op.GOES_WITH, f"goes_with(module, dependencies) — channel_b relation"),
                (Op.TOGETHER_ALONE, f"together(declarations) — module composition"),
            ],
            f_ops=[
                (Op.IF_THEN, f"if_then(precondition, postcondition) — code contract"),
                (Op.MUST_LET, f"must(type_safety) — effect annotation"),
            ],
        )

    return _build_example(
        seg,
        g_ops=[(Op.THIS, f"this(record) — T1 structured data")],
        s_ops=[
            (Op.GOES_WITH, f"goes_with(record, format) — channel_b relation"),
            (Op.INSIDE_OUTSIDE, f"inside(record, T1_corpus)"),
        ],
        f_ops=[(Op.SAME_NOT_SAME, f"same(reference, transformed) — equivalence check")],
    )


def _decompose_t2(seg: Segment) -> FrameExample:
    """T2: RTL design documents — Verilog modules, design specs."""
    text = seg.text

    # Module decomposition
    if seg.segment_type == "module":
        module_name = _extract_module_name(text)
        ports = _extract_ports(text)
        return _build_example(
            seg,
            g_ops=[
                (Op.THIS, f"this(module={module_name}) — design unit"),
                (Op.BECAUSE, f"because(RTL_specification, {module_name})"),
            ],
            s_ops=[
                (Op.INSIDE_OUTSIDE, f"inside({module_name}, design_hierarchy)"),
                (Op.TOGETHER_ALONE, f"together({', '.join(ports[:4])}) — port interface"),
            ],
            f_ops=[
                (Op.CAN_CANNOT, f"can({module_name}, synthesize)"),
                (Op.MUST_LET, f"must(timing_constraints) — channel_c requirement"),
            ],
        )

    # Section/paragraph decomposition — ensure home operators in each modality
    return _build_example(
        seg,
        g_ops=[(Op.THIS, f"this(section) — design document content")],
        s_ops=[
            (Op.GOES_WITH, f"goes_with(section, T2_document) — channel_b relation"),
            (Op.INSIDE_OUTSIDE, f"inside(content, document_scope)"),
        ],
        f_ops=[
            (Op.IF_THEN, f"if_then(requirement, behavior)"),
            (Op.NEAR_FAR, f"near(section, related_context) — document proximity"),
        ],
    )


def _decompose_chat(seg: Segment) -> FrameExample:
    """T3 chat messages — extract operator reasoning from conversation."""
    import re
    text = seg.text
    text_lower = text.lower()
    role = seg.segment_type.replace("chat_", "")

    # ── Ch.A: Channel A (Tier 1: THIS, SAME/NOT-SAME, NO) ──
    g_ops: list[tuple[Op, str]] = [
        (Op.THIS, f"this({role}_message) — deictic anchor to speaker turn"),
    ]
    # If the message references identity/equivalence
    if any(w in text_lower for w in ["same", "identical", "equal", "different", "not the same", "distinguish"]):
        g_ops.append((Op.SAME_NOT_SAME, f"same/not-same detected in: {text[:80]}"))
    # If the message uses negation channel_bly
    if any(w in text_lower for w in ["not ", "no ", "never ", "cannot ", "don't", "doesn't", "isn't"]):
        g_ops.append((Op.NO, f"no(negation) in: {text[:80]}"))

    # ── Ch.B: Channel B (Tier 2: GOES-WITH, TOGETHER/ALONE, MANY/ONE, EVERY/SOME, MORE/LESS, CAN/CANNOT) ──
    s_ops: list[tuple[Op, str]] = []

    # GOES-WITH: association/coupling language
    if any(w in text_lower for w in ["coupled", "associated", "related", "connected", "linked", "maps to", "corresponds"]):
        s_ops.append((Op.GOES_WITH, f"goes_with(relation) in: {text[:80]}"))
    else:
        # Default: message goes-with its conversation context
        s_ops.append((Op.GOES_WITH, f"goes_with({role}, conversation_context)"))

    # TOGETHER/ALONE: composition language
    if any(w in text_lower for w in ["together", "combined", "joint", "both", "separate", "alone", "independent", "isolated"]):
        s_ops.append((Op.TOGETHER_ALONE, f"together/alone in: {text[:80]}"))

    # MANY/ONE: cardinality language
    if any(w in text_lower for w in ["multiple", "several", "many", "each", "single", "one ", "unique"]):
        s_ops.append((Op.MANY_ONE, f"many/one in: {text[:80]}"))

    # EVERY/SOME: quantification
    if any(w in text_lower for w in ["every ", "all ", "any ", "some ", "each "]):
        s_ops.append((Op.EVERY_SOME, f"every/some in: {text[:80]}"))

    # MORE/LESS: comparative
    if any(w in text_lower for w in ["more ", "less ", "greater", "smaller", "better", "worse", "stronger", "weaker"]):
        s_ops.append((Op.MORE_LESS, f"more/less in: {text[:80]}"))

    # CAN/CANNOT: capability
    if any(w in text_lower for w in ["can ", "cannot ", "capable", "able to", "unable"]):
        s_ops.append((Op.CAN_CANNOT, f"can/cannot in: {text[:80]}"))

    # Ensure at least one Tier 2 operator
    if not any(op in _TIER_2_OPS for op, _ in s_ops):
        s_ops.insert(0, (Op.GOES_WITH, f"goes_with({role}, topic) — default channel_b relation"))

    # ── Ch.C: Channel C (Tier 3: INSIDE/OUTSIDE, NEAR/FAR, IF/THEN, BECAUSE, MAYBE, MUST/LET) ──
    f_ops: list[tuple[Op, str]] = []

    # IF/THEN: conditional reasoning
    if any(w in text_lower for w in ["if ", "then ", "when ", "implies", "conditional", "given that"]):
        f_ops.append((Op.IF_THEN, f"if_then in: {text[:80]}"))

    # BECAUSE: causal reasoning
    if any(w in text_lower for w in ["because", "since ", "therefore", "thus ", "hence", "reason", "caused by"]):
        f_ops.append((Op.BECAUSE, f"because in: {text[:80]}"))

    # MUST/LET: deontic
    if any(w in text_lower for w in ["must ", "shall ", "required", "permitted", "allowed", "forbidden"]):
        f_ops.append((Op.MUST_LET, f"must/let in: {text[:80]}"))

    # MAYBE: epistemic uncertainty
    if any(w in text_lower for w in ["maybe", "perhaps", "possibly", "might ", "uncertain", "unclear"]):
        f_ops.append((Op.MAYBE, f"maybe in: {text[:80]}"))

    # INSIDE/OUTSIDE: containment/boundary
    if any(w in text_lower for w in ["inside", "outside", "within", "boundary", "scope", "contained", "enclosed"]):
        f_ops.append((Op.INSIDE_OUTSIDE, f"inside/outside in: {text[:80]}"))

    # NEAR/FAR: proximity/distance language
    if any(w in text_lower for w in ["near ", "far ", "close to", "distant", "proxim", "remote", "adjacent", "approach"]):
        f_ops.append((Op.NEAR_FAR, f"near/far in: {text[:80]}"))

    # Default: at least one Tier 3 operator
    if not any(op in _CAUSAL_OPS for op, _ in f_ops):
        f_ops.append((Op.IF_THEN, f"if_then({role}_states, content_follows) — default channel_c"))

    return _build_example(seg, g_ops=g_ops, s_ops=s_ops, f_ops=f_ops)


# Operator sets for home-condition checking
_TIER_2_OPS = {Op.GOES_WITH, Op.TOGETHER_ALONE, Op.MANY_ONE, Op.EVERY_SOME, Op.MORE_LESS, Op.CAN_CANNOT}
_CAUSAL_OPS = {Op.INSIDE_OUTSIDE, Op.NEAR_FAR, Op.IF_THEN, Op.BECAUSE, Op.MAYBE, Op.MUST_LET}


def _decompose_t3(seg: Segment) -> FrameExample:
    """T3: Formal property specifications — SVA assertions, constraints."""
    text = seg.text

    if seg.segment_type == "assertion":
        return _build_example(
            seg,
            g_ops=[
                (Op.THIS, f"this(assertion) — formal property"),
                (Op.BECAUSE, f"because(design_requirement, property)"),
            ],
            s_ops=[
                (Op.EVERY_SOME, f"every(cycle, property_holds)"),
                (Op.MUST_LET, f"must(property) — invariant constraint"),
            ],
            f_ops=[
                (Op.IF_THEN, f"if_then(antecedent, consequent) — temporal implication"),
                (Op.SAME_NOT_SAME, f"same(design_intent, formal_property)"),
            ],
        )

    return _build_example(
        seg,
        g_ops=[(Op.THIS, f"this(formal_content) — T3 specification")],
        s_ops=[
            (Op.GOES_WITH, f"goes_with(property, verification_scope) — channel_b relation"),
            (Op.INSIDE_OUTSIDE, f"inside(property, formal_context)"),
        ],
        f_ops=[(Op.MUST_LET, f"must(constraint) — formal requirement")],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_example(
    seg: Segment,
    g_ops: list[tuple[Op, str]],
    s_ops: list[tuple[Op, str]],
    f_ops: list[tuple[Op, str]],
) -> FrameExample:
    """Construct a FrameExample from operator lists and auto-witness.

    Sorts each modality's operators by DAG index (non-decreasing)
    before building the sequence. This ensures G₁ channel_b integrity.
    """
    # Sort by operator index to maintain DAG ordering
    g_sorted = sorted(g_ops, key=lambda x: x[0].value)
    s_sorted = sorted(s_ops, key=lambda x: x[0].value)
    f_sorted = sorted(f_ops, key=lambda x: x[0].value)

    channel_a = ModalityGrounding(
        modality="G",
        operators=OperatorSequence(
            expressions=[OperatorExpression(operator=op, evidence=ev) for op, ev in g_sorted]
        ),
    )
    channel_b = ModalityGrounding(
        modality="S",
        operators=OperatorSequence(
            expressions=[OperatorExpression(operator=op, evidence=ev) for op, ev in s_sorted]
        ),
    )
    channel_c = ModalityGrounding(
        modality="F",
        operators=OperatorSequence(
            expressions=[OperatorExpression(operator=op, evidence=ev) for op, ev in f_sorted]
        ),
    )

    witnesses = _auto_witness(seg)

    provenance = SourceProvenance(
        source_id=seg.source.source_id,
        tier=seg.source.tier,
        url=seg.source.url,
        commit_or_version=seg.source.commit_or_version,
        license=seg.source.license,
        acquired_at=seg.source.acquired_at,
        artifact_sha256=seg.source.artifact_sha256,
    )

    example = FrameExample(
        provenance=provenance,
        channel_a=channel_a,
        channel_b=channel_b,
        channel_c=channel_c,
        witnesses=witnesses,
    )
    example.content_hash = example.compute_hash()
    return example


def _auto_witness(seg: Segment) -> WitnessBundle:
    """Generate witness attestations from segment metadata."""
    bundle = WitnessBundle()
    bundle.attestations[Witness.WHAT] = WitnessAttestation(
        witness=Witness.WHAT, attested=True,
        evidence=f"{seg.segment_type} segment, {len(seg.text)} chars",
    )
    bundle.attestations[Witness.WHERE] = WitnessAttestation(
        witness=Witness.WHERE, attested=True,
        evidence=f"source={seg.source.source_id}, bytes={seg.byte_range}",
    )
    bundle.attestations[Witness.WHICH] = WitnessAttestation(
        witness=Witness.WHICH, attested=True,
        evidence=f"tier={seg.source.tier.value}, type={seg.segment_type}",
    )
    bundle.attestations[Witness.WHEN] = WitnessAttestation(
        witness=Witness.WHEN, attested=True,
        evidence=f"acquired={seg.source.acquired_at}",
    )
    bundle.attestations[Witness.FOR_WHAT] = WitnessAttestation(
        witness=Witness.FOR_WHAT, attested=True,
        evidence="governed training data for governed generation pipeline",
    )
    bundle.attestations[Witness.HOW] = WitnessAttestation(
        witness=Witness.HOW, attested=True,
        evidence=f"pipeline S3 decompose, tier={seg.source.tier.value}",
    )
    bundle.attestations[Witness.WHENCE] = WitnessAttestation(
        witness=Witness.WHENCE, attested=True,
        evidence=f"url={seg.source.url}, commit={seg.source.commit_or_version}",
    )
    return bundle


def _extract_module_name(text: str) -> str:
    import re
    m = re.search(r"module\s+(\w+)", text)
    return m.group(1) if m else "unknown"


def _extract_ports(text: str) -> list[str]:
    import re
    return re.findall(r"\b(?:input|output|inout)\s+(?:\[\d+:\d+\]\s*)?(\w+)", text)
