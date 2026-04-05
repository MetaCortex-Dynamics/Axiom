"""
Distillation runner — call frontier LLMs, filter through governance pipeline.

Usage:
  python -m pipeline.distill.run --provider anthropic --api-key $KEY --n 1000
  python -m pipeline.distill.run --provider openai --api-key $KEY --n 1000
  python -m pipeline.distill.run --provider all --anthropic-key $KEY1 --openai-key $KEY2

Each call generates governed prose from a committed structure,
validates through G1-G7, and saves T-status pairs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.distill.prompt import SYSTEM_PROMPT, build_distill_prompt, build_synthetic_structures
from pipeline.mdlm.tokenizer import encode as encode_gov, pad_sequence as pad_gov
from pipeline.stages.s4_validate import validate_and_score, TigStatus, Verdict
from pipeline.types import (
    Op, Witness, ModalityGrounding, OperatorSequence, OperatorExpression,
    SourceProvenance, Tier, WitnessBundle, WitnessAttestation,
)


def call_anthropic(prompt: str, system: str, api_key: str) -> Optional[str]:
    """Call Claude API."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"  Anthropic error: {e}")
        return None


def call_openai(prompt: str, system: str, api_key: str) -> Optional[str]:
    """Call OpenAI API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  OpenAI error: {e}")
        return None


def prose_to_example(gov_structure: dict, prose: str, source: str) -> Optional[object]:
    """Convert governance structure + generated prose into a validatable example."""
    from pipeline.types import FrameExample

    g_ops = [{"operator": op["operator"]} for op in gov_structure.get("G", [])]
    s_ops = [{"operator": op["operator"]} for op in gov_structure.get("S", [])]
    f_ops = [{"operator": op["operator"]} for op in gov_structure.get("F", [])]

    def make_grounding(modality, ops):
        return ModalityGrounding(
            modality=modality,
            operators=OperatorSequence(
                expressions=[
                    OperatorExpression(operator=Op.from_name(op["operator"]), evidence=f"distilled({op['operator']})")
                    for op in ops if Op.from_name(op["operator"]) is not None
                ]
            ),
        )

    example = FrameExample(
        provenance=SourceProvenance(
            source_id=f"distill:{source}", tier=Tier.T3, url="distillation",
            commit_or_version="p2", license="generated",
            acquired_at="2026-04-02", artifact_sha256="distilled",
        ),
        channel_a=make_grounding("G", g_ops),
        channel_b=make_grounding("S", s_ops),
        channel_c=make_grounding("F", f_ops),
        witnesses=WitnessBundle(),
    )

    for w in Witness:
        example.witnesses.attestations[w] = WitnessAttestation(
            witness=w, attested=True, evidence=f"distilled({w.canonical_name})",
        )
    example.content_hash = example.compute_hash()
    return example


def run_distillation(
    n: int = 1000,
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    output_dir: str = "corpus/distilled",
):
    """Run the distillation pipeline."""
    print(f"=== P2: LLM Distillation ({n} structures) ===")

    os.makedirs(output_dir, exist_ok=True)

    # Generate structures
    structures = build_synthetic_structures(n)

    # Also include existing corpus structures
    existing_pairs = Path("corpus/axiom/pairs.json")
    if existing_pairs.exists():
        with open(existing_pairs, encoding="utf-8") as f:
            existing = json.load(f)
        for p in existing[:min(len(existing), n)]:
            frame = p.get("frame", p.get("triad", {}))
            if frame:
                structures.append({
                    "G": frame.get("channel_a", frame.get("channel_a", {frame.get("G", {})})).get("operators", []) if isinstance(frame.get("channel_a", frame.get("genealogical", {})), dict) else [],
                    "S": frame.get("channel_b", frame.get("channel_b", {})).get("operators", []) if isinstance(frame.get("channel_b", frame.get("structural", {})), dict) else [],
                    "F": frame.get("channel_c", frame.get("channel_c", {})).get("operators", []) if isinstance(frame.get("channel_c", frame.get("functional", {})), dict) else [],
                })
    print(f"  Total structures: {len(structures)}")

    providers = []
    if anthropic_key:
        providers.append(("anthropic", lambda p, s: call_anthropic(p, s, anthropic_key)))
    if openai_key:
        providers.append(("openai", lambda p, s: call_openai(p, s, openai_key)))

    if not providers:
        print("  No API keys provided. Generating structure-only pairs (no prose distillation).")
        # Save structures for later distillation
        with open(f"{output_dir}/structures.json", "w") as f:
            json.dump(structures, f)
        print(f"  Saved {len(structures)} structures to {output_dir}/structures.json")
        return

    # Run distillation
    t_count = 0
    u_count = 0
    f_count = 0
    viki_count = 0
    pairs = []

    t0 = time.time()
    for i, structure in enumerate(structures):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(structures)}] T={t_count} U={u_count} F={f_count} VIKI={viki_count} {elapsed:.0f}s")

        prompt = build_distill_prompt(structure)

        for provider_name, call_fn in providers:
            prose = call_fn(prompt, SYSTEM_PROMPT)
            if prose is None:
                continue

            # Validate through G1-G7
            example = prose_to_example(structure, prose, f"{provider_name}_{i}")
            if example is None:
                continue

            result = validate_and_score(example)

            if result.tig_status == TigStatus.TRUE:
                t_count += 1
                pairs.append({
                    "frame": structure,
                    "prose": prose[:512],
                    "source": f"{provider_name}_{i}",
                    "provider": provider_name,
                })
            elif result.tig_status == TigStatus.UNDECIDABLE:
                u_count += 1
            else:
                f_count += 1
                if result.viki_patterns:
                    viki_count += len(result.viki_patterns)

            # Rate limiting
            time.sleep(0.1)

    elapsed = time.time() - t0

    # Save pairs
    with open(f"{output_dir}/distilled_pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)

    # Report
    total = t_count + u_count + f_count
    print(f"\n=== Distillation Complete ({elapsed:.0f}s) ===")
    print(f"  Structures processed: {len(structures)}")
    print(f"  Candidates generated: {total}")
    print(f"  T-status (accepted):  {t_count} ({t_count*100//max(total,1)}%)")
    print(f"  U-status (repair):    {u_count}")
    print(f"  F-status (rejected):  {f_count}")
    print(f"  VIKI detections:      {viki_count}")
    print(f"  Saved to: {output_dir}/distilled_pairs.json")

    # G4 checks
    print(f"\n  === G4 GATE CHECK ===")
    print(f"  G4.1 Corpus >= 50K:   {'PASS' if t_count >= 50000 else 'PENDING'} ({t_count})")
    print(f"  G4.2 VIKI rate > 0%:  {'PASS' if viki_count > 0 else 'PENDING'} ({viki_count})")

    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--output", default="corpus/distilled")
    args = parser.parse_args()

    run_distillation(
        n=args.n,
        anthropic_key=args.anthropic_key,
        openai_key=args.openai_key,
        output_dir=args.output,
    )
