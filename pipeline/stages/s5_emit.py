"""
S5: EMIT — Write validated FrameExamples to CORPUS-SEMI-001/.

Per SPEC-PIPELINE-001 Part B.2:
  Format: One JSONL file per source, each line a complete FrameExample.
  Corpus-level manifest.json with aggregate statistics.
  Provenance chain: source → segment → classification → decomposition →
    validation → crystallinity → emission timestamp → example SHA-256.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Sequence

from pipeline.stages.s4_validate import ValidationResult, Verdict
from pipeline.types import ExclusionRecord, FrameExample


def emit(
    examples: Sequence[tuple[FrameExample, ValidationResult]],
    exclusions: Sequence[ExclusionRecord],
    output_dir: str,
) -> str:
    """Emit validated examples and exclusion manifest to output directory.

    Returns path to manifest.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Group examples by source
    by_source: dict[str, list[dict]] = {}
    total_pass = 0
    total_oracle = 0
    total_fail = 0

    for example, result in examples:
        if result.verdict == Verdict.FAIL:
            total_fail += 1
            continue
        if result.verdict == Verdict.ORACLE_QUEUE:
            total_oracle += 1
            continue

        total_pass += 1
        source_id = example.provenance.source_id
        if source_id not in by_source:
            by_source[source_id] = []

        record = {
            "content_hash": example.content_hash,
            "provenance": asdict(example.provenance),
            "channel_a": _grounding_to_dict(example.channel_a),
            "channel_b": _grounding_to_dict(example.channel_b),
            "channel_c": _grounding_to_dict(example.channel_c),
            "witnesses": {
                w.canonical_name: {"attested": a.attested, "evidence": a.evidence}
                for w, a in example.witnesses.attestations.items()
            },
            "crystallinity": result.crystallinity_score,
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }
        by_source[source_id].append(record)

    # Write JSONL files
    source_files = []
    for source_id, records in by_source.items():
        safe_name = source_id.replace("/", "_").replace(":", "_")[:80]
        filename = f"{safe_name}.jsonl"
        filepath = os.path.join(examples_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        source_files.append({
            "source_id": source_id,
            "filename": filename,
            "example_count": len(records),
        })

    # Write exclusion manifest
    exclusion_path = os.path.join(output_dir, "exclusion_manifest.jsonl")
    with open(exclusion_path, "w", encoding="utf-8") as f:
        for exc in exclusions:
            f.write(json.dumps(asdict(exc), ensure_ascii=False) + "\n")

    # Write corpus manifest
    manifest = {
        "spec": "SPEC-PIPELINE-001 v0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "total_pass": total_pass,
            "total_oracle_queue": total_oracle,
            "total_fail": total_fail,
            "total_exclusions": len(exclusions),
            "source_count": len(by_source),
        },
        "sources": source_files,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


def _grounding_to_dict(grounding) -> dict:
    return {
        "modality": grounding.modality,
        "operators": [
            {
                "operator": expr.operator.canonical_name,
                "arguments": expr.arguments,
                "evidence": expr.evidence,
            }
            for expr in grounding.operators.expressions
        ],
    }
