"""
MDLM Tokenizer — Encodes governed structures as discrete token sequences.

The MDLM learns the STRUCTURE of valid operator compositions, not the
prose content. Evidence strings are metadata for traceability — they are
NOT tokenized. The kernel learns which operators appear in which modalities
in which order, with which witness attestations.

Vocabulary (~32 tokens):
  - 15 operator tokens (THIS through NEAR/FAR)
  - 7 witness tokens (WHAT through WHENCE)
  - 2 witness status tokens (ATTESTED, WITHHELD)
  - 6 channel_b delimiters (<G> </G> <S> </S> <F> </F>)
  - 2 sequence tokens (<BOS> <EOS>)
  - 2 special tokens (<PAD> <MASK>)

Total: 34 tokens. Orders of magnitude smaller than prose LLM vocabularies.
The complexity lives in sequence-level structure, not token identity.

Sequence format:
  <BOS> <G> op op op </G> <S> op op </S> <F> op op </F>
         WIT:A WIT:A WIT:A WIT:A WIT:A WIT:A WIT:A <EOS>

hierarchical masking tiers:
  Tier 1 (Tier 1):       THIS, SAME/NOT-SAME, NO
  Tier 2 (Tier 2):     GOES-WITH, TOGETHER/ALONE, MANY/ONE, EVERY/SOME, MORE/LESS, CAN/CANNOT
  Tier 3 (Tier 3 + readiness): INSIDE/OUTSIDE, NEAR/FAR, IF/THEN, BECAUSE,
          MAYBE, MUST/LET, + witness status tokens
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.types import Op, Witness


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

# Special tokens
PAD   = 0
MASK  = 1
BOS   = 2
EOS   = 3

# Channel B delimiters
G_OPEN  = 4
G_CLOSE = 5
S_OPEN  = 6
S_CLOSE = 7
F_OPEN  = 8
F_CLOSE = 9

# 15 operator tokens (indices 10-24, matching Op enum + 10)
OP_OFFSET = 10

# 7 witness tokens (indices 25-31)
WIT_OFFSET = 25

# Witness status
ATTESTED = 32
WITHHELD = 33

VOCAB_SIZE = 34

# Token names for display
TOKEN_NAMES = [
    "<PAD>", "<MASK>", "<BOS>", "<EOS>",
    "<G>", "</G>", "<S>", "</S>", "<F>", "</F>",
    "THIS", "GOES-WITH", "MANY/ONE", "EVERY/SOME", "NO",
    "IF/THEN", "BECAUSE", "SAME/NOT-SAME", "INSIDE/OUTSIDE",
    "CAN/CANNOT", "MAYBE", "MUST/LET", "TOGETHER/ALONE",
    "MORE/LESS", "NEAR/FAR",
    "WHAT", "WHERE", "WHICH", "WHEN", "FOR-WHAT", "HOW", "WHENCE",
    "ATTESTED", "WITHHELD",
]

assert len(TOKEN_NAMES) == VOCAB_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# hierarchical MASKING TIERS
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1: Tier 1 (3 operators) — unmasked first
TIER_1_TOKENS = {
    OP_OFFSET + Op.THIS,
    OP_OFFSET + Op.SAME_NOT_SAME,
    OP_OFFSET + Op.NO,
}

# Tier 2: Tier 2 (6 operators) — unmasked second
TIER_2_TOKENS = {
    OP_OFFSET + Op.GOES_WITH,
    OP_OFFSET + Op.TOGETHER_ALONE,
    OP_OFFSET + Op.MANY_ONE,
    OP_OFFSET + Op.EVERY_SOME,
    OP_OFFSET + Op.MORE_LESS,
    OP_OFFSET + Op.CAN_CANNOT,
}

# Tier 3: Tier 3 (6 operators) + witness status (9 total) — unmasked last
TIER_3_TOKENS = {
    OP_OFFSET + Op.INSIDE_OUTSIDE,
    OP_OFFSET + Op.NEAR_FAR,
    OP_OFFSET + Op.IF_THEN,
    OP_OFFSET + Op.BECAUSE,
    OP_OFFSET + Op.MAYBE,
    OP_OFFSET + Op.MUST_LET,
    ATTESTED,
    WITHHELD,
    # Witness identity tokens are also Tier 3 (readiness readiness)
} | {WIT_OFFSET + w for w in Witness}

# Channel B tokens are never masked — they define the frame
NEVER_MASKED = {PAD, BOS, EOS, G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE}


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODE / DECODE
# ═══════════════════════════════════════════════════════════════════════════════

def encode(example: dict) -> list[int]:
    """Encode a FrameExample (from JSONL) as a token sequence.

    Format:
      <BOS> <G> op... </G> <S> op... </S> <F> op... </F>
             wit:status wit:status ... <EOS>
    """
    tokens = [BOS]

    # Modalities
    for mod_key, open_tok, close_tok in [
        ("channel_a", G_OPEN, G_CLOSE),
        ("channel_b", S_OPEN, S_CLOSE),
        ("channel_c", F_OPEN, F_CLOSE),
    ]:
        tokens.append(open_tok)
        mod = example.get(mod_key, {})
        for op_entry in mod.get("operators", []):
            op_name = op_entry.get("operator", "")
            op_val = Op.from_name(op_name)
            if op_val is not None:
                tokens.append(OP_OFFSET + op_val.value)
        tokens.append(close_tok)

    # Witnesses
    for w in Witness:
        wit_data = example.get("witnesses", {}).get(w.canonical_name, {})
        tokens.append(WIT_OFFSET + w.value)
        if wit_data.get("attested", False):
            tokens.append(ATTESTED)
        else:
            tokens.append(WITHHELD)

    tokens.append(EOS)
    return tokens


def decode(tokens: list[int]) -> str:
    """Decode a token sequence to human-readable string."""
    return " ".join(TOKEN_NAMES[t] if 0 <= t < VOCAB_SIZE else f"?{t}" for t in tokens)


def pad_sequence(tokens: list[int], max_len: int) -> list[int]:
    """Pad or truncate a token sequence to fixed length."""
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [PAD] * (max_len - len(tokens))


def get_tier(token_id: int) -> int:
    """Return the masking tier for a token (1, 2, 3, or 0 for never-masked)."""
    if token_id in NEVER_MASKED:
        return 0
    if token_id in TIER_1_TOKENS:
        return 1
    if token_id in TIER_2_TOKENS:
        return 2
    if token_id in TIER_3_TOKENS:
        return 3
    return 0  # unknown tokens are channel_b


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(corpus_dir: str | Path) -> list[list[int]]:
    """Load all governed examples from a corpus directory and encode them."""
    corpus_dir = Path(corpus_dir)
    examples_dir = corpus_dir / "examples"
    if not examples_dir.exists():
        examples_dir = corpus_dir

    sequences = []
    for jsonl_path in sorted(examples_dir.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                tokens = encode(example)
                sequences.append(tokens)

    return sequences


def corpus_statistics(sequences: list[list[int]]) -> dict:
    """Compute statistics over encoded corpus."""
    from collections import Counter

    lengths = [len(s) for s in sequences]
    token_counts = Counter()
    tier_counts = Counter()

    for seq in sequences:
        for t in seq:
            token_counts[t] += 1
            tier_counts[get_tier(t)] += 1

    return {
        "num_sequences": len(sequences),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0,
        "vocab_usage": {TOKEN_NAMES[t]: c for t, c in token_counts.most_common()},
        "tier_distribution": {f"tier_{t}": c for t, c in sorted(tier_counts.items())},
    }
