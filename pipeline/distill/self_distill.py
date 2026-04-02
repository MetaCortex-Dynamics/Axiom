"""
Self-distillation — Axiom generates its own training data.

PROPOSE (MDLM) → structures
EXECUTE (500M decoder) → candidate prose
DECIDE (G1-G7) → filter
T-status → expanded corpus

No external LLM. No API cost. No IP exposure.
The admissibility engine is the filter.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from pipeline.mdlm.tokenizer import (
    VOCAB_SIZE as STRUCT_VOCAB, PAD as STRUCT_PAD,
    encode as encode_gov, pad_sequence as pad_gov,
    TOKEN_NAMES, OP_OFFSET,
    G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE,
    WIT_OFFSET, ATTESTED,
)
from pipeline.mdlm.model import StructureModel, MaskingSchedule, generate as mdlm_generate
from pipeline.mdlm.governed_pipeline import propose, decide, promote, tokens_to_example
from pipeline.stages.s4_validate import validate_and_score, TigStatus
from pipeline.distill.prompt import build_synthetic_structures


def run_self_distillation(
    mdlm_path: str = "models/axiom/mdlm_best.pt",
    decoder_path: str = "models/axiom-500m/decoder_best.pt",
    tokenizer_path: str = "models/axiom/bpe_tokenizer.json",
    n_rounds: int = 10,
    candidates_per_round: int = 200,
    output_dir: str = "corpus/distilled",
    device_name: str = "cuda",
):
    """Self-distillation: Axiom generates and filters its own training data."""
    print(f"=== Self-Distillation: {n_rounds} rounds x {candidates_per_round} candidates ===")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Load MDLM (PROPOSE)
    mdlm = StructureModel(vocab_size=STRUCT_VOCAB, d_model=128, nhead=4, num_layers=4, max_len=40).to(device)
    mdlm.load_state_dict(torch.load(mdlm_path, weights_only=True, map_location=device))

    # Load BPE tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    bpe_vocab = tokenizer.get_vocab_size()
    BPE_BOS = tokenizer.token_to_id("<bos>")
    BPE_EOS = tokenizer.token_to_id("<eos>")

    # Load 500M decoder (EXECUTE) — try to load, fall back to 12.9M
    try:
        from pipeline.mdlm.decoder_500m import Axiom500MDecoder
        decoder = Axiom500MDecoder(
            struct_vocab=STRUCT_VOCAB, prose_vocab=bpe_vocab,
            d_model=1024, nhead=16, num_encoder_layers=6, num_decoder_layers=24,
            max_struct_len=40, max_prose_len=256, use_checkpoint=False,
        ).to(device)
        state = torch.load(decoder_path, weights_only=True, map_location=device)
        decoder.load_state_dict(state)
        decoder.eval()
        print(f"  Loaded 500M decoder from {decoder_path}")
        use_500m = True
    except Exception as e:
        print(f"  500M decoder not available ({e}), using 12.9M")
        from pipeline.mdlm.decoder import ConstrainedDecoder
        decoder = ConstrainedDecoder(
            gov_vocab=STRUCT_VOCAB, prose_vocab=bpe_vocab,
            d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=6,
            max_struct_len=40, max_prose_len=128,
        ).to(device)
        state = torch.load("models/axiom/decoder_best.pt", weights_only=True, map_location=device)
        state = {k.replace("triad_embedding", "struct_embedding").replace("triad_pos", "struct_pos"): v for k, v in state.items()}
        decoder.load_state_dict(state)
        decoder.eval()
        use_500m = False

    # Self-distillation loop
    all_pairs = []
    total_t = 0
    total_f = 0
    total_viki = 0
    t0 = time.time()

    for round_idx in range(n_rounds):
        # Phase 1: PROPOSE — generate structures
        candidates = propose(mdlm, num_candidates=candidates_per_round, g_slots=2, s_slots=2, f_slots=2)

        # Phase 2: DECIDE — validate structures
        decided = decide(candidates)
        admitted = [(c, d, e) for c, d, e in decided if d.tig_status == "T" and e is not None]

        # Phase 3: PROMOTE
        promoted = promote(admitted)

        round_t = 0
        round_f = 0

        for example, commitment in promoted:
            # Phase 4: EXECUTE — generate prose from committed structure
            gov_dict = {
                "channel_a": {"operators": [{"operator": e.operator.canonical_name, "evidence": e.evidence} for e in example.channel_a.operators.expressions]},
                "channel_b": {"operators": [{"operator": e.operator.canonical_name, "evidence": e.evidence} for e in example.channel_b.operators.expressions]},
                "channel_c": {"operators": [{"operator": e.operator.canonical_name, "evidence": e.evidence} for e in example.channel_c.operators.expressions]},
                "witnesses": {w.canonical_name: {"attested": a.attested, "evidence": a.evidence} for w, a in example.witnesses.attestations.items()},
            }

            tt = torch.tensor([pad_gov(encode_gov(gov_dict), 40)], dtype=torch.long, device=device)

            # Generate prose
            if use_500m:
                gen_ids = decoder.generate(tt, BPE_BOS, BPE_EOS, max_len=200, temperature=0.8)
                prose = tokenizer.decode(gen_ids[0])
            else:
                struct_h = decoder.struct_embedding(tt) + decoder.struct_pos(torch.arange(40, device=device).unsqueeze(0))
                mem = decoder.encoder(struct_h, src_key_padding_mask=(tt == STRUCT_PAD))
                ids = torch.tensor([[BPE_BOS]], dtype=torch.long, device=device)
                gen = []
                with torch.no_grad():
                    for _ in range(120):
                        ph = decoder.prose_embedding(ids) + decoder.prose_pos(torch.arange(ids.size(1), device=device).unsqueeze(0))
                        dec = decoder.decoder(ph, mem,
                            tgt_mask=nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=device),
                            memory_key_padding_mask=(tt == STRUCT_PAD))
                        nxt = torch.multinomial(F.softmax(decoder.output_proj(dec[:, -1, :]) / 0.8, dim=-1), 1)
                        ids = torch.cat([ids, nxt], dim=1)
                        if nxt.item() == BPE_EOS:
                            break
                        gen.append(nxt.item())
                prose = tokenizer.decode(gen)

            if not prose.strip():
                round_f += 1
                continue

            # Re-validate the (structure, prose) pair
            result = validate_and_score(example)
            if result.tig_status == TigStatus.TRUE:
                round_t += 1
                all_pairs.append({
                    "frame": {
                        "channel_a": gov_dict["channel_a"],
                        "channel_b": gov_dict["channel_b"],
                        "channel_c": gov_dict["channel_c"],
                    },
                    "prose": prose[:512],
                    "source": f"self_distill_r{round_idx}",
                })
            else:
                round_f += 1
                if hasattr(result, 'viki_patterns') and result.viki_patterns:
                    total_viki += len(result.viki_patterns)

        total_t += round_t
        total_f += round_f
        elapsed = time.time() - t0
        print(f"  Round {round_idx+1}/{n_rounds}: +{round_t} T-status, {round_f} rejected, total={total_t} ({elapsed:.0f}s)")

    # Save
    output_path = f"{output_dir}/self_distilled_pairs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n=== Self-Distillation Complete ({elapsed:.0f}s) ===")
    print(f"  Rounds: {n_rounds}")
    print(f"  T-status pairs: {total_t}")
    print(f"  Rejected: {total_f}")
    print(f"  VIKI detections: {total_viki}")
    print(f"  Saved to: {output_path}")

    return all_pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--output", default="corpus/distilled")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_self_distillation(
        n_rounds=args.rounds,
        candidates_per_round=args.candidates,
        output_dir=args.output,
        device_name=args.device,
    )
