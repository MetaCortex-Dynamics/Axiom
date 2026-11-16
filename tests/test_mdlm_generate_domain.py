"""QG-SP-001 generation repair R1 (successor of adopted QG-SP-001-GEN-ABD @ d35d873).

Answers Dot 0043 against the adopted bytes:
  INPUT_DOMAIN        T = 0 raised ZeroDivisionError.
  INVALID_PREDICTIONS always-MASK logits ended as PAD 0 in every slot.
  FALLBACK            step 0 and a conditional extra pass could commit anything.
All rules below are NEW sampler choices (labels D1-D3, V0-V3, F1-F3); none is
paper canon. Band order, thresholds and FLAT readiness tier 2 stay as adopted.
"""

import random

import pytest

from pipeline.mdlm.tokenizer import (
    ATTESTED, BOS, EOS, F_CLOSE, F_OPEN, G_CLOSE, G_OPEN, MASK, OP_OFFSET, PAD,
    S_CLOSE, S_OPEN, VOCAB_SIZE, WITHHELD, WIT_OFFSET,
)
from pipeline.types import Op

torch = pytest.importorskip("torch")

from pipeline.mdlm.model import MaskingSchedule, generate, schedule_tier  # noqa: E402

T = 9
SEQ_LEN = 40
G_POS, S_POS, F_POS = [2, 3, 4], [7, 8, 9, 10], [13, 14, 15]
OP_POS = G_POS + S_POS + F_POS
W_POS = [18 + 2 * k for k in range(7)]
SLOTS = OP_POS + W_POS
OPS = set(range(OP_OFFSET, OP_OFFSET + 15))
STATUS = {ATTESTED, WITHHELD}
ALL = list(MaskingSchedule)
TIERED = [MaskingSchedule.HIERARCHICAL, MaskingSchedule.FLAT, MaskingSchedule.INVERTED]
FRAME = ([BOS, G_OPEN, MASK, MASK, MASK, G_CLOSE, S_OPEN, MASK, MASK, MASK, MASK, S_CLOSE,
          F_OPEN, MASK, MASK, MASK, F_CLOSE]
         + sum(([WIT_OFFSET + w, MASK] for w in range(7)), []) + [EOS])


class _FnStub(torch.nn.Module):
    """fn(position, step) -> token id, or None for a MASK-only preference."""

    def __init__(self, fn):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.fn = fn
        self.calls = []

    def forward(self, x, t):
        step = int(t[0])
        self.calls.append((x.clone(), step))
        B, L = x.shape
        logits = torch.full((B, L, VOCAB_SIZE), -1e9)
        logits[:, :, MASK] = 0.0
        for i in range(L):
            tok = self.fn(i, step)
            if tok is not None:
                logits[:, i, :] = -1e9
                logits[:, i, tok] = 0.0
        return logits


def _run(fn, schedule, total=T, **kw):
    stub = _FnStub(fn)
    out = generate(stub, 1, SEQ_LEN, schedule, total, 3, 4, 3, **kw)
    return stub, out


def _commit(stub, i):
    return min(t for x, t in stub.calls if x[0, i].item() == MASK)


def _assert_valid(out):
    row = out[0].tolist()
    for i in OP_POS:
        assert row[i] in OPS, (i, row[i])
    for i in W_POS:
        assert row[i] in STATUS, (i, row[i])
    for i, tok in enumerate(FRAME):
        if tok != MASK:
            assert row[i] == tok, (i, row[i])
    assert all(tok == PAD for tok in row[len(FRAME):])


# ── D1-D3 input domain ────────────────────────────────────────────────────

@pytest.mark.parametrize("schedule", ALL)
@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_total_timesteps_outside_domain_is_rejected(schedule, bad):
    with pytest.raises(ValueError):
        _run(lambda i, s: None, schedule, total=bad)


@pytest.mark.parametrize("schedule", ALL)
def test_total_timesteps_one_is_the_smallest_run(schedule):
    stub, out = _run(lambda i, s: None, schedule, total=1)
    assert [t for _, t in stub.calls] == [1, 0]
    _assert_valid(out)


@pytest.mark.parametrize("kw", [
    dict(num_samples=0), dict(num_samples=-1),
    dict(seq_len=len(FRAME) - 1),
    dict(g_slots=-1), dict(s_slots=-1), dict(f_slots=-1),
])
def test_shape_arguments_outside_domain_are_rejected(kw):
    args = dict(num_samples=1, seq_len=SEQ_LEN, g_slots=3, s_slots=4, f_slots=3)
    args.update(kw)
    with pytest.raises(ValueError):
        generate(_FnStub(lambda i, s: None), args["num_samples"], args["seq_len"],
                 MaskingSchedule.HIERARCHICAL, T, args["g_slots"], args["s_slots"],
                 args["f_slots"])


def test_seq_len_equal_to_frame_keeps_eos():
    stub = _FnStub(lambda i, s: None)
    out = generate(stub, 1, len(FRAME), MaskingSchedule.HIERARCHICAL, T, 3, 4, 3)
    assert out[0, -1].item() == EOS


# ── V0-V3 invalid predictions ─────────────────────────────────────────────

@pytest.mark.parametrize("schedule", ALL)
def test_always_mask_logits_give_a_valid_sequence(schedule):
    stub, out = _run(lambda i, s: None, schedule)
    _assert_valid(out)
    assert len(stub.calls) == T + 1


@pytest.mark.parametrize("schedule", ALL)
@pytest.mark.parametrize("invalid_op,invalid_w", [
    (PAD, PAD), (BOS, EOS), (G_OPEN, F_CLOSE), (ATTESTED, OP_OFFSET + Op.THIS),
    (WIT_OFFSET, WIT_OFFSET + 3), (MASK, MASK),
])
def test_out_of_slot_predictions_are_never_committed(schedule, invalid_op, invalid_w):
    def fn(i, step):
        if i in OP_POS:
            return invalid_op
        return invalid_w if i in W_POS else None
    torch.manual_seed(0)
    stub, out = _run(fn, schedule)
    _assert_valid(out)
    assert len(stub.calls) == T + 1


@pytest.mark.parametrize("schedule", TIERED)
def test_valid_preference_wins_over_invalid_mass(schedule):
    tier_tok = {1: OP_OFFSET + Op.THIS, 2: OP_OFFSET + Op.GOES_WITH, 3: OP_OFFSET + Op.IF_THEN}

    class _Mixed(_FnStub):
        def forward(self, x, t):
            logits = super().forward(x, t)
            for i in OP_POS:
                logits[:, i, :] = -1e9
                logits[:, i, PAD] = 5.0       # invalid, highest
                logits[:, i, tier_tok[1 + OP_POS.index(i) % 3]] = 0.0
            for i in W_POS:
                logits[:, i, :] = -1e9
                logits[:, i, G_OPEN] = 5.0    # invalid, highest
                logits[:, i, WITHHELD] = 0.0
            return logits
    stub = _Mixed(lambda i, s: None)
    out = generate(stub, 1, SEQ_LEN, schedule, T, 3, 4, 3)
    _assert_valid(out)
    for i in OP_POS:
        assert out[0, i].item() == tier_tok[1 + OP_POS.index(i) % 3]
    assert all(out[0, w].item() == WITHHELD for w in W_POS)


# ── F1-F3 fallback contract ───────────────────────────────────────────────

def _band(tier, step, total, schedule):
    if schedule == MaskingSchedule.INVERTED and tier in (1, 3):
        tier = 4 - tier
    frac = step / total
    return (tier == 1 and frac >= 0.66) or (tier == 2 and 0.33 <= frac < 0.66) or \
        (tier == 3 and frac < 0.33)


@pytest.mark.parametrize("schedule", TIERED)
@pytest.mark.parametrize("seed", range(5))
def test_random_changing_logits_commit_in_band_or_at_step_zero(schedule, seed):
    rng = random.Random(seed)
    table = {}

    def fn(i, step):
        if i not in SLOTS:
            return None
        key = (i, step)
        if key not in table:
            pool = sorted(STATUS) if i in W_POS else sorted(OPS)
            table[key] = rng.choice(pool + [None, None])
        return table[key]
    torch.manual_seed(seed)
    stub, out = _run(fn, schedule, total=20)
    _assert_valid(out)
    for i in SLOTS:
        step = _commit(stub, i)
        tok = out[0, i].item()
        if step > 0:
            assert _band(schedule_tier(tok, schedule), step, 20, schedule)
        for x, t in stub.calls:
            if t < step:
                assert x[0, i].item() == tok


def test_step_zero_fallback_is_not_global_tier_order():
    """Pins the non-guarantee: a tier-1 slot that abstains until step 0 commits
    after tier-3 slots. NOT-SAME(band-or-step-0, global tier order)."""
    t1, t3 = OP_OFFSET + Op.THIS, OP_OFFSET + Op.IF_THEN

    def fn(i, step):
        if i == G_POS[0]:
            return t1 if step == 0 else None
        if i in F_POS:
            return t3
        return None
    stub, out = _run(fn, MaskingSchedule.HIERARCHICAL)
    assert out[0, G_POS[0]].item() == t1
    assert _commit(stub, G_POS[0]) == 0 < _commit(stub, F_POS[0])


@pytest.mark.parametrize("schedule", ALL)
def test_no_extra_pass_model_calls_are_exactly_t_plus_one(schedule):
    def fn(i, step):
        return MASK if i in SLOTS else None   # MASK as an explicit prediction
    stub, out = _run(fn, schedule)
    assert [t for _, t in stub.calls] == list(range(T, -1, -1))
    _assert_valid(out)
