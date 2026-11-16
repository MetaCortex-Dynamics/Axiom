"""QG-SP-001 linked generation obligation: generate() must honour the schedule.

Reverse-time contract recovered from Axiom@c16ef87 bytes:
  tokenizer.py:90,97,107  tier 1 unmasked first, tier 2 second, tier 3 last
  model.py:11             A: Tier 1 -> Tier 2 -> Tier 3+readiness
  model.py:143-144        tier 1 masked last, unmasked first; tier 3 the reverse
  model.py:45,162         D (INVERTED): tier 1 masked first, so unmasked last
generate() counts steps down from T, so the tier unmasked first has the
highest commit step. UNIFORM and the sampling budget are not pinned here:
no Axiom byte fixes a per-step rule for them.

A stub model predicts a fixed token per slot, so each slot's commit step is
deterministic. The commit step of a slot is the timestep of the last forward
call that still saw it masked.
"""

import pytest

from pipeline.mdlm.tokenizer import ATTESTED, MASK, OP_OFFSET, VOCAB_SIZE
from pipeline.types import Op

torch = pytest.importorskip("torch")

from pipeline.mdlm.model import MaskingSchedule, generate  # noqa: E402

TIER_1 = OP_OFFSET + Op.THIS
TIER_2 = OP_OFFSET + Op.GOES_WITH
TIER_3 = OP_OFFSET + Op.IF_THEN
TOTAL_TIMESTEPS = 9
SEQ_LEN = 40
G_SLOTS, S_SLOTS, F_SLOTS = 3, 4, 3


class _Stub(torch.nn.Module):
    """G slots predict tier 1, S slots tier 2, F slots tier 3, witness slots ATTESTED."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, x, t):
        self.calls.append((x.clone(), int(t[0])))
        B, L = x.shape
        logits = torch.full((B, L, VOCAB_SIZE), -1e9)
        g0 = 2
        s0 = g0 + G_SLOTS + 2
        f0 = s0 + S_SLOTS + 2
        w0 = f0 + F_SLOTS + 1
        for i in range(L):
            if g0 <= i < g0 + G_SLOTS:
                tok = TIER_1
            elif s0 <= i < s0 + S_SLOTS:
                tok = TIER_2
            elif f0 <= i < f0 + F_SLOTS:
                tok = TIER_3
            elif i >= w0 and (i - w0) % 2 == 1 and i < w0 + 14:
                tok = ATTESTED
            else:
                continue
            logits[:, i, tok] = 0.0
        return logits


def _commit_steps(schedule):
    """Return {predicted token: set of commit steps} for one generation."""
    stub = _Stub()
    out = generate(stub, 1, SEQ_LEN, schedule, TOTAL_TIMESTEPS,
                   G_SLOTS, S_SLOTS, F_SLOTS)
    first = stub.calls[0][0][0]
    slots = [i for i in range(SEQ_LEN) if first[i].item() == MASK]
    steps = {}
    for i in slots:
        last_masked = min(t for x, t in stub.calls if x[0, i].item() == MASK)
        steps.setdefault(out[0, i].item(), set()).add(last_masked)
    return steps


def _step(steps, tok):
    assert len(steps[tok]) == 1, steps
    return next(iter(steps[tok]))


def _order(s):
    return _step(s, TIER_1), _step(s, TIER_2), _step(s, TIER_3)


def test_stub_predicts_every_slot():
    steps = _commit_steps(MaskingSchedule.HIERARCHICAL)
    assert set(steps) == {TIER_1, TIER_2, TIER_3, ATTESTED}


def test_hierarchical_unmasks_tier_1_first_and_tier_3_last():
    t1, t2, t3 = _order(_commit_steps(MaskingSchedule.HIERARCHICAL))
    assert t1 > t2 > t3


def test_hierarchical_readiness_commits_with_tier_3():
    s = _commit_steps(MaskingSchedule.HIERARCHICAL)
    assert _step(s, ATTESTED) == _step(s, TIER_3) != _step(s, TIER_2)


def test_flat_unmasks_operators_in_hierarchical_order():
    t1, t2, t3 = _order(_commit_steps(MaskingSchedule.FLAT))
    assert t1 > t2 > t3


def test_flat_readiness_commits_with_tier_2():
    s = _commit_steps(MaskingSchedule.FLAT)
    assert _step(s, ATTESTED) == _step(s, TIER_2) != _step(s, TIER_3)


def test_inverted_unmasks_tier_3_first_and_tier_1_last():
    t1, t2, t3 = _order(_commit_steps(MaskingSchedule.INVERTED))
    assert t3 > t2 > t1


def test_inverted_readiness_commits_with_tier_3():
    s = _commit_steps(MaskingSchedule.INVERTED)
    assert _step(s, ATTESTED) == _step(s, TIER_3) != _step(s, TIER_2)


# ── Extended obligation: readiness, invalid predictions, changing logits,
#    timestep boundaries, common budget. Bands follow the recovered contract:
#    tier 1 in t_frac >= 0.66, tier 2 in [0.33, 0.66), tier 3 below 0.33
#    (A and B); D swaps tiers 1 and 3. Step 0 commits every remaining slot.

from pipeline.mdlm.tokenizer import (  # noqa: E402
    G_OPEN, PAD, TIER_1_TOKENS, TIER_2_TOKENS, TIER_3_TOKENS, WITHHELD,
)
from pipeline.mdlm.model import schedule_tier  # noqa: E402

TIERED = [MaskingSchedule.HIERARCHICAL, MaskingSchedule.FLAT, MaskingSchedule.INVERTED]
ALL = TIERED + [MaskingSchedule.UNIFORM]
G_POS, S_POS, F_POS = [2, 3, 4], [7, 8, 9, 10], [13, 14, 15]
W_POS = [18 + 2 * k for k in range(7)]


class _FnStub(torch.nn.Module):
    """Predicts fn(position, step) per slot; None means no preference (MASK)."""

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


def _run(fn, schedule, total=TOTAL_TIMESTEPS):
    stub = _FnStub(fn)
    out = generate(stub, 1, SEQ_LEN, schedule, total, G_SLOTS, S_SLOTS, F_SLOTS)
    return stub, out


def _commit(stub, i):
    """Step at which slot i was committed (last step that saw it masked)."""
    return min(t for x, t in stub.calls if x[0, i].item() == MASK)


def _band(tier, step, total, schedule):
    if schedule == MaskingSchedule.INVERTED and tier in (1, 3):
        tier = 4 - tier
    frac = step / total
    return (tier == 1 and frac >= 0.66) or (tier == 2 and 0.33 <= frac < 0.66) or \
        (tier == 3 and frac < 0.33)


@pytest.mark.parametrize("ready", [ATTESTED, WITHHELD])
@pytest.mark.parametrize("schedule,with_tier", [
    (MaskingSchedule.HIERARCHICAL, TIER_3),
    (MaskingSchedule.FLAT, TIER_2),
    (MaskingSchedule.INVERTED, TIER_3),
])
def test_readiness_status_commits_with_its_schedule_tier(ready, schedule, with_tier):
    def fn(i, step):
        if i in W_POS:
            return ready
        if i in S_POS:
            return TIER_2
        if i in F_POS:
            return TIER_3
        return TIER_1 if i in G_POS else None
    stub, out = _run(fn, schedule)
    ref = (S_POS if with_tier == TIER_2 else F_POS)[0]
    other = (F_POS if with_tier == TIER_2 else S_POS)[0]
    assert {_commit(stub, w) for w in W_POS} == {_commit(stub, ref)}
    assert _commit(stub, ref) != _commit(stub, other)
    assert all(out[0, w].item() == ready for w in W_POS)


@pytest.mark.parametrize("schedule", ALL)
def test_mask_prediction_retries_and_never_survives(schedule):
    def fn(i, step):
        if i == G_POS[0]:
            return None              # always MASK
        if i == G_POS[1]:
            return TIER_1 if step <= 4 else None
        return TIER_1 if i in G_POS + S_POS + F_POS else (ATTESTED if i in W_POS else None)
    stub, out = _run(fn, schedule)
    assert MASK not in out[0].tolist()
    assert _commit(stub, G_POS[1]) <= 4


@pytest.mark.parametrize("schedule", TIERED)
@pytest.mark.parametrize("invalid", [PAD, G_OPEN])
def test_invalid_tier_0_prediction_waits_for_final_step(schedule, invalid):
    def fn(i, step):
        return invalid if i == G_POS[0] else (TIER_1 if i in G_POS + S_POS + F_POS else
                                              (ATTESTED if i in W_POS else None))
    stub, _ = _run(fn, schedule)
    assert _commit(stub, G_POS[0]) == 0


@pytest.mark.parametrize("schedule", TIERED)
def test_changing_logits_commit_only_inside_open_band(schedule):
    cycle = [TIER_1, TIER_2, TIER_3]

    def fn(i, step):
        if i in G_POS + S_POS + F_POS:
            return cycle[(i + step) % 3]
        return [ATTESTED, WITHHELD][(i + step) % 2] if i in W_POS else None
    stub, out = _run(fn, schedule)
    slots = G_POS + S_POS + F_POS + W_POS
    for i in slots:
        step = _commit(stub, i)
        tok = out[0, i].item()
        assert step == 0 or _band(schedule_tier(tok, schedule), step, TOTAL_TIMESTEPS, schedule)
        # committed values never change on later calls
        for x, t in stub.calls:
            if t < step:
                assert x[0, i].item() == tok


@pytest.mark.parametrize("schedule,token,first_valid,expected", [
    (MaskingSchedule.HIERARCHICAL, TIER_1, 66, 66),
    (MaskingSchedule.HIERARCHICAL, TIER_1, 65, 0),
    (MaskingSchedule.HIERARCHICAL, TIER_2, 66, 65),
    (MaskingSchedule.HIERARCHICAL, TIER_2, 33, 33),
    (MaskingSchedule.HIERARCHICAL, TIER_3, 33, 32),
    (MaskingSchedule.FLAT, ATTESTED, 33, 33),
    (MaskingSchedule.FLAT, ATTESTED, 32, 0),
    (MaskingSchedule.INVERTED, TIER_3, 66, 66),
    (MaskingSchedule.INVERTED, TIER_1, 33, 32),
])
def test_timestep_band_boundaries(schedule, token, first_valid, expected):
    slot = W_POS[0] if token in (ATTESTED, WITHHELD) else G_POS[0]

    def fn(i, step):
        if i == slot:
            return token if step <= first_valid else None
        return None
    stub, out = _run(fn, schedule, total=100)
    assert _commit(stub, slot) == expected
    assert out[0, slot].item() == token


@pytest.mark.parametrize("schedule", ALL)
def test_common_budget_t_plus_one_model_calls(schedule):
    def fn(i, step):
        return TIER_1 if i in G_POS + S_POS + F_POS else (ATTESTED if i in W_POS else None)
    stub, out = _run(fn, schedule)
    assert len(stub.calls) == TOTAL_TIMESTEPS + 1
    assert MASK not in out[0].tolist()
