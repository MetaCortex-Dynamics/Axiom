# PLAN-AXIOM-001: Gated Release

## Phase Structure

```
P1 ──→ P2 ──→ P3 ──→ P4 ──→ P5
SCALE  DIST   BENCH  SUITE  LAUNCH
 G3     G4     G5     G6     G7
```

---

## P1: Decoder Scale-Up

[THIS] 500M decoder. 24 layers, 16 heads, 1024 hidden, 8192 BPE.
[GOES-WITH] Frozen MDLM encoder (934K) via cross-attention.
[TOGETHER] L = L_ce + a1·L_telos + a2·L_witness + a3·L_audit.

### G3

| # | [MUST/LET] | Predicate | Threshold |
|---|-----------|-----------|-----------|
| 3.1 | MUST | reconstruction loss | < 0.05 |
| 3.2 | MUST | fluency (human, n=50) | >= 3/5 |
| 3.3 | LET | audit pass rate | >= 90% |
| 3.4 | LET | FOR-WHAT alignment | >= 85% |
| 3.5 | MUST | governance trace preserved | 100% |

[IF/THEN] G3 PASS → P2 unblocked.
[IF/THEN] G3.2 trending but not met → proceed to P2 (data volume is the fix).

---

## P2: LLM Distillation

[THIS] 100K+ training pairs from frontier LLM oracles.
[EVERY] committed structure → [3 LLMs] → [EVERY candidate through G1-G7].
[INSIDE] T-status → corpus. [OUTSIDE] F-status → rejected with BECAUSE.

### Volume

| Source | Inputs | x3 LLMs | Est. T-status |
|--------|--------|---------|---------------|
| MetaCortex archive | 4,432 | 13,296 | ~8-10K |
| Semiconductor corpus | 10,435 | 31,305 | ~20-25K |
| Synthetic compositions | 20,000 | 60,000 | ~40-50K |
| **TOTAL** | 34,867 | 104,601 | **~70-85K** |

### G4

| # | [MUST/LET] | Predicate | Threshold |
|---|-----------|-----------|-----------|
| 4.1 | MUST | corpus size | >= 50K T-status |
| 4.2 | MUST | VIKI rejection rate | > 0% (gate must filter) |
| 4.3 | MUST | fluency improvement over G3 | > 10% |
| 4.4 | LET | [EVERY] LLM contributes T-status | all 3 |
| 4.5 | LET | [NO] single source > 60% | balanced |

[IF/THEN] G4 PASS → P3 unblocked.

---

## P3: Benchmark Validation

[THIS] competitive scores on domain + reasoning benchmarks.
[SAME/NOT-SAME] Axiom 500M vs Phi-3 3.8B, Gemma 2B, TinyLlama 1.1B.

### Targets

| Benchmark | [IF/THEN] operator mapping | Threshold |
|-----------|---------------------------|-----------|
| CEC accuracy | IF/THEN + SAME + INSIDE | >= industry baseline |
| GSM8K | IF/THEN + BECAUSE + MORE/LESS + EVERY | >= 40% |
| TruthfulQA | WHENCE + NO | > same-class average |
| Safety | [EVERY] operator + [EVERY] witness | > [EVERY] comparison |
| HumanEval | IF/THEN + INSIDE + MUST + CAN | >= 20% |

### G5

| # | [MUST/LET] | Predicate | Threshold |
|---|-----------|-----------|-----------|
| 5.1 | MUST | CEC accuracy | >= baseline |
| 5.2 | MUST | GSM8K | >= 40% |
| 5.3 | MUST | TruthfulQA | > same-class |
| 5.4 | MUST | Safety | > [EVERY] comparison |
| 5.5 | LET | HumanEval | >= 20% |

[IF/THEN] G5 PASS → P4 unblocked.

---

## P4: Governance Benchmark Suite

[THIS] the benchmark [NO] other model [CAN] pass.
[SAME/NOT-SAME] Axiom scores > 0% where [EVERY] LLM scores 0%.

### Metrics

| Metric | [MUST] Axiom target | [EVERY] LLM score |
|--------|--------------------|--------------------|
| Structural validity | >= 99% | 0% |
| Witness attestation | >= 98% | 0% |
| VIKI contamination | 0% | unmeasured |
| Output traceability | 100% | 0% |
| Trace completeness | 100% | 0% |
| Jailbreak resistance | 100% | < 100% |

### G6

| # | [MUST/LET] | Predicate | Threshold |
|---|-----------|-----------|-----------|
| 6.1 | MUST | suite discriminates | Axiom > 0% where LLMs = 0% |
| 6.2 | MUST | reproducible | third-party [CAN] run independently |
| 6.3 | MUST | Axiom hits targets | [EVERY] target in metrics met |
| 6.4 | MUST | publishable | [NO] protected IP exposed |

[IF/THEN] G6 PASS → P5 unblocked.

---

## P5: Package and Launch

[THIS] public release.
[TOGETHER] GitHub + YouTube + phone demo + benchmark paper.

### Components

| Component | License | [INSIDE/OUTSIDE] moat |
|-----------|---------|----------------------|
| Reference GLM (12.9M) | MIT | OUTSIDE (open) |
| Theorem papers | CC-BY-4.0 | OUTSIDE (theory) |
| Governance benchmark | MIT | OUTSIDE (theory) |
| Production Axiom (500M+) | Commercial | INSIDE (product) |
| Decomposition pipeline | Proprietary | INSIDE (moat) |
| Distillation pipeline | Proprietary | INSIDE (moat) |
| Domain corpora | Proprietary | INSIDE (fuel) |

### G7

| # | [MUST/LET] | Predicate | Threshold |
|---|-----------|-----------|-----------|
| 7.1 | MUST | phone inference | < 1s, < 50MB |
| 7.2 | MUST | benchmarks published | G5 + G6 in paper |
| 7.3 | MUST | GitHub live | code + papers + suite |
| 7.4 | MUST | launch video | >= 1 with phone demo |
| 7.5 | MUST | Invariant 1 sweep | [NO] IP in public materials |
| 7.6 | LET | commercial pipeline | production deployable |

[IF/THEN] G7 PASS → LAUNCH.

---

## Critical Path

```
P1(2w) → P2(2w) → P3(1w) → P4(1w) → P5(2w) = 8 weeks
```

[BECAUSE] P1 is the binding constraint. [EVERY] downstream phase depends on decoder quality.
[MAYBE] P1 underfits on 4,432 pairs → P2 distillation is the fix.
[NEAR] P4 + P5 [CAN] overlap with P3.

---

## Minimum Viable Launch

G3 + G5.1 + G5.4 + G6.1 + G7.1 + G7.3 + G7.5

[MUST] Reference GLM on GitHub.
[MUST] Phone demo working.
[MUST] CEC benchmark competitive.
[MUST] Safety benchmark dominant.
[MUST] Governance suite discriminates.
[MUST] Invariant 1 clean.

## Market Impact (30 days post-launch)

[SOME] of: >1K GitHub stars, tech press names "governed language model", inbound from regulated industry.

---

## Next Action

[THIS] Train 500M BPE decoder with protocol loss terms on 4,432 pairs.
[WHEN] Now.
[FOR-WHAT] G3 evaluation at end of Week 2.
