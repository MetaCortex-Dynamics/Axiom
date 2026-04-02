"""Smoke tests for pipeline types — verify parity with kernel definitions."""

from pipeline.types import (
    Classifier, ExclusionReason, ExclusionRecord, ModalityGrounding, Op,
    OperatorExpression, OperatorSequence, OracleDecision, OracleReviewItem,
    OracleVerdict, PipelineStage, SourceProvenance, Tier, FrameExample,
    Witness, WitnessAttestation, WitnessBundle,
)


def test_operator_count():
    assert len(Op) == 15


def test_operator_dag_order():
    for i, op in enumerate(Op):
        assert op.value == i


def test_operator_canonical_names():
    assert Op.THIS.canonical_name == "THIS"
    assert Op.GOES_WITH.canonical_name == "GOES-WITH"
    assert Op.NEAR_FAR.canonical_name == "NEAR/FAR"


def test_operator_roundtrip():
    for op in Op:
        assert Op.from_name(op.canonical_name) == op


def test_operator_rejects_folk():
    assert Op.from_name("WHO") is None
    assert Op.from_name("WHY") is None


def test_witness_count():
    assert len(Witness) == 7


def test_witness_canonical_names():
    assert Witness.FOR_WHAT.canonical_name == "FOR-WHAT"
    assert Witness.WHENCE.canonical_name == "WHENCE"


def test_witness_bundle_unanimity():
    bundle = WitnessBundle()
    for w in Witness:
        bundle.attestations[w] = WitnessAttestation(
            witness=w, attested=True, evidence="test"
        )
    assert bundle.is_unanimous()


def test_witness_bundle_withheld_blocks():
    bundle = WitnessBundle()
    for w in Witness:
        bundle.attestations[w] = WitnessAttestation(
            witness=w, attested=(w != Witness.FOR_WHAT), evidence="test"
        )
    assert not bundle.is_unanimous()


def test_operator_sequence_ordering():
    seq = OperatorSequence(expressions=[
        OperatorExpression(operator=Op.THIS, evidence="x"),
        OperatorExpression(operator=Op.BECAUSE, evidence="y"),
        OperatorExpression(operator=Op.NEAR_FAR, evidence="z"),
    ])
    assert seq.verify_ordering()


def test_operator_sequence_violation():
    seq = OperatorSequence(expressions=[
        OperatorExpression(operator=Op.NEAR_FAR, evidence="z"),
        OperatorExpression(operator=Op.THIS, evidence="x"),
    ])
    assert not seq.verify_ordering()


def test_gov_example_hash_deterministic():
    prov = SourceProvenance(
        source_id="test", tier=Tier.T1, url="", commit_or_version="",
        license="MIT", acquired_at="2026-03-27", artifact_sha256="abc",
    )
    ex1 = FrameExample(
        provenance=prov,
        channel_a=ModalityGrounding(modality="G", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.THIS, evidence="a")]
        )),
        channel_b=ModalityGrounding(modality="S", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.INSIDE_OUTSIDE, evidence="b")]
        )),
        channel_c=ModalityGrounding(modality="F", operators=OperatorSequence(
            expressions=[OperatorExpression(operator=Op.IF_THEN, evidence="c")]
        )),
        witnesses=WitnessBundle(),
    )
    ex1.content_hash = ex1.compute_hash()
    ex2 = FrameExample(
        provenance=prov,
        channel_a=ex1.channel_a,
        channel_b=ex1.channel_b,
        channel_c=ex1.channel_c,
        witnesses=WitnessBundle(),
    )
    ex2.content_hash = ex2.compute_hash()
    assert ex1.content_hash == ex2.content_hash


def test_exclusion_reasons():
    assert len(ExclusionReason) == 8
    assert ExclusionReason.UNTRANSLATABLE == "Untranslatable"


def test_tiers():
    assert len(Tier) == 3
    assert Tier.T1.value == "T1"
