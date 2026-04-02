"""
Phase 5: Constrained Decoder — EXECUTE phase of the GGP.

Takes a committed governed structure (from PROMOTE) and generates
natural language within the validity envelope.

Architecture: Small transformer decoder conditioned on governed operator
tokens. The governed structure is the prompt; the output is prose that
expresses the structure in natural language.

This is NOT a general-purpose LLM. It generates governed prose —
text whose semantic content is constrained to what the governed permits.
The decoder cannot introduce implicit authority structures because
the governed frame doesn't encode them.

Training data: (structure tokens, source text) pairs extracted from
the decomposition pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pipeline.mdlm.tokenizer import (
    encode as encode_gov, VOCAB_SIZE as STRUCT_VOCAB_SIZE,
    BOS, EOS, PAD, TOKEN_NAMES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAIRED DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameProsePair:
    """A (governed structure, source prose) pair for decoder training."""
    gov_tokens: list[int]  # Encoded governed structure
    prose: str               # Original source text
    source_id: str


def extract_pairs_from_pipeline(
    corpus_dir: str | Path,
    theory_dir: str | Path | None = None,
) -> list[FrameProsePair]:
    """Extract structure-prose pairs by re-running the pipeline with text capture.

    Since the emitted JSONL doesn't store the original text, we re-run
    the decomposition and capture both the governed and the source segment.
    """
    import sys
    sys.path.insert(0, ".")

    from pipeline.ingest.chat_archive import ingest_conversation_file
    from pipeline.stages.s2_classify import classify, Classification
    from pipeline.stages.s3_decompose import decompose
    from pipeline.stages.s4_validate import validate_and_score, TigStatus, Verdict

    pairs = []

    if theory_dir:
        theory_path = Path(theory_dir)
        for conv_file in sorted(theory_path.glob("conv_*.json")):
            try:
                for seg in ingest_conversation_file(conv_file):
                    c = classify(seg)
                    if c.classification != Classification.TECHNICAL:
                        continue
                    ex = decompose(c)
                    if ex is None:
                        continue
                    r = validate_and_score(ex)
                    if r.tig_status != TigStatus.TRUE:
                        continue

                    # Build pair
                    struct_dict = {
                        "channel_a": {"operators": [
                            {"operator": e.operator.canonical_name, "evidence": e.evidence}
                            for e in ex.channel_a.operators.expressions
                        ]},
                        "channel_b": {"operators": [
                            {"operator": e.operator.canonical_name, "evidence": e.evidence}
                            for e in ex.channel_b.operators.expressions
                        ]},
                        "channel_c": {"operators": [
                            {"operator": e.operator.canonical_name, "evidence": e.evidence}
                            for e in ex.channel_c.operators.expressions
                        ]},
                        "witnesses": {
                            w.canonical_name: {"attested": a.attested, "evidence": a.evidence}
                            for w, a in ex.witnesses.attestations.items()
                        },
                    }
                    gov_tokens = encode_gov(struct_dict)

                    pairs.append(FrameProsePair(
                        gov_tokens=gov_tokens,
                        prose=seg.text[:512],  # Cap at 512 chars for training
                        source_id=ex.provenance.source_id,
                    ))
            except Exception:
                continue

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE TOKENIZER (character-level for simplicity)
# ═══════════════════════════════════════════════════════════════════════════════

PROSE_PAD = 0
PROSE_BOS = 1
PROSE_EOS = 2
PROSE_UNK = 3
PROSE_VOCAB_OFFSET = 4

# Build vocab from printable ASCII + common unicode
PROSE_CHARS = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
PROSE_VOCAB_SIZE = PROSE_VOCAB_OFFSET + len(PROSE_CHARS)
_CHAR_TO_ID = {c: i + PROSE_VOCAB_OFFSET for i, c in enumerate(PROSE_CHARS)}


def encode_prose(text: str, max_len: int = 256) -> list[int]:
    """Encode prose as character-level token IDs."""
    tokens = [PROSE_BOS]
    for ch in text[:max_len - 2]:
        tokens.append(_CHAR_TO_ID.get(ch, PROSE_UNK))
    tokens.append(PROSE_EOS)
    return tokens


def decode_prose(token_ids: list[int]) -> str:
    """Decode character-level token IDs back to text."""
    id_to_char = {v: k for k, v in _CHAR_TO_ID.items()}
    chars = []
    for tid in token_ids:
        if tid in (PROSE_PAD, PROSE_BOS, PROSE_EOS):
            continue
        if tid == PROSE_UNK:
            chars.append("?")
        else:
            chars.append(id_to_char.get(tid, "?"))
    return "".join(chars)


def pad_prose(tokens: list[int], max_len: int) -> list[int]:
    """Pad or truncate prose tokens to fixed length."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [PROSE_PAD] * (max_len - len(tokens))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINED DECODER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class ConstrainedDecoder(nn.Module):
        """Transformer decoder conditioned on governed structure.

        Encoder: processes governed token sequence (the committed structure)
        Decoder: generates prose character-by-character within the envelope

        The structure tokens serve as cross-attention keys — the decoder
        can only attend to the committed structure, not to arbitrary context.
        """

        def __init__(
            self,
            gov_vocab: int = STRUCT_VOCAB_SIZE,
            prose_vocab: int = PROSE_VOCAB_SIZE,
            d_model: int = 128,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            num_decoder_layers: int = 4,
            max_struct_len: int = 40,
            max_prose_len: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.max_prose_len = max_prose_len

            # Encoder (governed structure)
            self.struct_embedding = nn.Embedding(gov_vocab, d_model, padding_idx=PAD)
            self.struct_pos = nn.Embedding(max_struct_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            # Decoder (prose generation)
            self.prose_embedding = nn.Embedding(prose_vocab, d_model, padding_idx=PROSE_PAD)
            self.prose_pos = nn.Embedding(max_prose_len, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
            self.output_proj = nn.Linear(d_model, prose_vocab)

        def forward(
            self,
            gov_tokens: torch.Tensor,   # (B, struct_len)
            prose_tokens: torch.Tensor,    # (B, prose_len)
        ) -> torch.Tensor:
            """Forward pass. Returns logits (B, prose_len, prose_vocab)."""
            B = gov_tokens.size(0)

            # Encode governed structure
            struct_len = gov_tokens.size(1)
            struct_pos = torch.arange(struct_len, device=gov_tokens.device).unsqueeze(0).expand(B, -1)
            struct_h = self.struct_embedding(gov_tokens) + self.struct_pos(struct_pos)
            struct_pad_mask = (gov_tokens == PAD)
            memory = self.encoder(struct_h, src_key_padding_mask=struct_pad_mask)

            # Decode prose
            prose_len = prose_tokens.size(1)
            prose_pos = torch.arange(prose_len, device=prose_tokens.device).unsqueeze(0).expand(B, -1)
            prose_h = self.prose_embedding(prose_tokens) + self.prose_pos(prose_pos)

            # Causal mask for autoregressive generation
            causal_mask = nn.Transformer.generate_square_subsequent_mask(prose_len, device=prose_tokens.device)
            prose_pad_mask = (prose_tokens == PROSE_PAD)

            decoded = self.decoder(
                prose_h, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=prose_pad_mask,
                memory_key_padding_mask=struct_pad_mask,
            )
            return self.output_proj(decoded)

        def generate(
            self,
            gov_tokens: torch.Tensor,  # (B, struct_len)
            max_len: int = 200,
            temperature: float = 0.8,
        ) -> list[str]:
            """Generate prose from governed structure."""
            self.eval()
            B = gov_tokens.size(0)
            device = gov_tokens.device

            # Encode governed
            struct_len = gov_tokens.size(1)
            struct_pos = torch.arange(struct_len, device=device).unsqueeze(0).expand(B, -1)
            struct_h = self.struct_embedding(gov_tokens) + self.struct_pos(struct_pos)
            struct_pad_mask = (gov_tokens == PAD)
            memory = self.encoder(struct_h, src_key_padding_mask=struct_pad_mask)

            # Autoregressive generation
            generated = torch.full((B, 1), PROSE_BOS, dtype=torch.long, device=device)

            with torch.no_grad():
                for _ in range(max_len):
                    prose_len = generated.size(1)
                    prose_pos = torch.arange(prose_len, device=device).unsqueeze(0).expand(B, -1)
                    prose_h = self.prose_embedding(generated) + self.prose_pos(prose_pos)
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(prose_len, device=device)

                    decoded = self.decoder(prose_h, memory, tgt_mask=causal_mask, memory_key_padding_mask=struct_pad_mask)
                    logits = self.output_proj(decoded[:, -1, :]) / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token], dim=1)

                    if (next_token == PROSE_EOS).all():
                        break

            results = []
            for b in range(B):
                results.append(decode_prose(generated[b].tolist()))
            return results
