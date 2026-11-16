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
