"""QG-SP-001 regression: the FLAT (3-6-6) schedule must not equal HIERARCHICAL (3-6-9).

Readiness tokens are the 7 witness-identity tokens plus ATTESTED and WITHHELD.
HIERARCHICAL keeps them in tier 3. FLAT demotes them to tier 2. Operator tiers
are the same in both schedules.
"""

import pytest

from pipeline.mdlm.tokenizer import (
    ATTESTED, BOS, EOS, F_CLOSE, F_OPEN, G_CLOSE, G_OPEN, OP_OFFSET,
    S_CLOSE, S_OPEN, TIER_1_TOKENS, TIER_2_TOKENS, TIER_3_TOKENS, WIT_OFFSET,
    WITHHELD,
)
from pipeline.types import Op, Witness

torch = pytest.importorskip("torch")

from pipeline.mdlm.model import MaskingSchedule, apply_mask  # noqa: E402

READINESS = {WIT_OFFSET + w for w in Witness} | {ATTESTED, WITHHELD}
TIER_3_OPERATORS = TIER_3_TOKENS - READINESS

# At t_frac = 0.33 with mask_rate = 1.0 the per-tier probabilities are exact:
# tier 1 -> 0.0, tier 2 -> 0.0, tier 3 -> 1.0. The masks are deterministic.
TIMESTEP = 33
TOTAL_TIMESTEPS = 100
MASK_RATE = 1.0


def _batch(rows: int = 4) -> "torch.Tensor":
    seq = [BOS,
           G_OPEN, OP_OFFSET + Op.THIS, OP_OFFSET + Op.NO, G_CLOSE,
           S_OPEN, OP_OFFSET + Op.GOES_WITH, OP_OFFSET + Op.CAN_CANNOT, S_CLOSE,
           F_OPEN, OP_OFFSET + Op.IF_THEN, OP_OFFSET + Op.BECAUSE, F_CLOSE]
    for w in Witness:
        seq += [WIT_OFFSET + w, ATTESTED if w % 2 == 0 else WITHHELD]
    seq.append(EOS)
    return torch.tensor([seq] * rows, dtype=torch.long)


def _columns(batch: "torch.Tensor", tokens: set[int]) -> list[int]:
    return [i for i, t in enumerate(batch[0].tolist()) if t in tokens]


def _mask(schedule: MaskingSchedule, batch: "torch.Tensor") -> "torch.Tensor":
    _, positions = apply_mask(batch, MASK_RATE, schedule, TIMESTEP, TOTAL_TIMESTEPS)
    return positions


def test_readiness_set_is_nine_tier_3_tokens():
    assert len(READINESS) == 9
    assert READINESS <= TIER_3_TOKENS
    assert len(TIER_3_OPERATORS) == 6


def test_flat_mask_differs_from_hierarchical():
    batch = _batch()
    assert not torch.equal(_mask(MaskingSchedule.FLAT, batch),
                           _mask(MaskingSchedule.HIERARCHICAL, batch))


def test_readiness_tokens_are_tier_3_under_hierarchical():
    batch = _batch()
    positions = _mask(MaskingSchedule.HIERARCHICAL, batch)
    assert positions[:, _columns(batch, READINESS)].all()


def test_readiness_tokens_are_tier_2_under_flat():
    batch = _batch()
    positions = _mask(MaskingSchedule.FLAT, batch)
    assert not positions[:, _columns(batch, READINESS)].any()


@pytest.mark.parametrize("schedule", [MaskingSchedule.HIERARCHICAL, MaskingSchedule.FLAT])
def test_operator_tiers_are_shared(schedule):
    batch = _batch()
    positions = _mask(schedule, batch)
    assert positions[:, _columns(batch, TIER_3_OPERATORS)].all()
    assert not positions[:, _columns(batch, TIER_1_TOKENS | TIER_2_TOKENS)].any()


def test_schedule_tier_assignment():
    from pipeline.mdlm.model import schedule_tier

    for tok in READINESS:
        assert schedule_tier(tok, MaskingSchedule.HIERARCHICAL) == 3
        assert schedule_tier(tok, MaskingSchedule.FLAT) == 2
    for schedule in MaskingSchedule:
        for tok in TIER_1_TOKENS:
            assert schedule_tier(tok, schedule) == 1
        for tok in TIER_2_TOKENS:
            assert schedule_tier(tok, schedule) == 2
        for tok in TIER_3_OPERATORS:
            assert schedule_tier(tok, schedule) == 3
