"""Integration tests for pipeline stages S1-S4."""

from pipeline.stages.s1_segment import Segment, segment
from pipeline.stages.s2_classify import Classification, classify
from pipeline.stages.s3_decompose import decompose
from pipeline.stages.s4_validate import Verdict, validate_and_score
from pipeline.types import SourceProvenance, Tier


def _t2_source() -> SourceProvenance:
    return SourceProvenance(
        source_id="test/counter.v",
        tier=Tier.T2,
        url="https://opencores.org/test",
        commit_or_version="abc123",
        license="LGPL-2.1",
        acquired_at="2026-03-27T00:00:00Z",
        artifact_sha256="deadbeef",
    )


VERILOG_MODULE = """\
module counter(
    input wire clk,
    input wire rst,
    output reg [7:0] count
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 8'b0;
        else
            count <= count + 1;
    end
endmodule
"""


def test_s1_segments_verilog_module():
    source = _t2_source()
    segs = list(segment(source, VERILOG_MODULE.encode()))
    module_segs = [s for s in segs if s.segment_type == "module"]
    assert len(module_segs) >= 1
    assert "counter" in module_segs[0].text


def test_s2_classifies_verilog_as_technical():
    source = _t2_source()
    segs = list(segment(source, VERILOG_MODULE.encode()))
    module_seg = [s for s in segs if s.segment_type == "module"][0]
    result = classify(module_seg)
    assert result.classification == Classification.TECHNICAL


def test_s2_classifies_license_as_nontechnical():
    source = _t2_source()
    seg = Segment(
        source=source,
        byte_range=(0, 100),
        text="Copyright 2024 OpenCores. All rights reserved. License: LGPL.",
        segment_type="section",
    )
    result = classify(seg)
    assert result.classification == Classification.NON_TECHNICAL


def test_s3_decomposes_verilog_module():
    source = _t2_source()
    segs = list(segment(source, VERILOG_MODULE.encode()))
    module_seg = [s for s in segs if s.segment_type == "module"][0]
    classified = classify(module_seg)
    example = decompose(classified)
    assert example is not None
    assert example.channel_a.operators.expressions
    assert example.channel_b.operators.expressions
    assert example.channel_c.operators.expressions


def test_s4_validates_decomposed_module():
    source = _t2_source()
    segs = list(segment(source, VERILOG_MODULE.encode()))
    module_seg = [s for s in segs if s.segment_type == "module"][0]
    classified = classify(module_seg)
    example = decompose(classified)
    assert example is not None
    result = validate_and_score(example)
    assert result.verdict != Verdict.FAIL, f"Validation failed: {result.errors}"


def test_full_pipeline_s1_through_s4():
    """End-to-end: segment -> classify -> decompose -> validate."""
    source = _t2_source()
    content = VERILOG_MODULE.encode()

    passed = []
    for seg in segment(source, content):
        classified = classify(seg)
        if classified.classification != Classification.TECHNICAL:
            continue
        example = decompose(classified)
        if example is None:
            continue
        result = validate_and_score(example)
        if result.verdict == Verdict.PASS:
            passed.append((example, result))

    assert len(passed) >= 1, "At least one module should pass full pipeline"
