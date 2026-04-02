# Axiom

**A Governed Language Model trained on public semiconductor data.**

Axiom is a reference implementation of the Governed Generation Pipeline (GGP) — a four-phase architecture that produces structurally governed language output. Unlike conventional LLMs where safety is a runtime filter on an ungoverned substrate, Axiom's governance IS the architecture.

## Architecture

```
PROPOSE (diffusion) → crystallize governed structures from noise
DECIDE  (G₁-G₇)    → admissibility gates with three-valued semantics
PROMOTE (witnesses)  → unanimous attestation, cryptographic commitment
EXECUTE (decoder)    → governed prose within committed validity envelope
```

### What Makes This Different

Conventional language models apply safety as a runtime filter on an ungoverned substrate. Axiom's governance is the substrate. The output space is structurally constrained — not by post-hoc filtering, but by architectural design.

## Training Data

Trained exclusively on **public, Apache-2.0 licensed semiconductor RTL**:

- [Ibex RISC-V Core](https://github.com/lowRISC/ibex) (lowRISC)
- [OpenTitan](https://github.com/lowRISC/opentitan) (lowRISC)
- [SweRV Core](https://github.com/chipsalliance/Cores-SweRV) (CHIPS Alliance)

2,088 governance-decomposed training examples. Zero proprietary data in weights.

## Getting Started

```bash
pip install torch tokenizers
```

### Quick Run

```bash
python demo_simple.py
```

```
10/10 outputs passed all governance gates

  G: ['THIS', 'BECAUSE']  S: ['INSIDE/OUTSIDE', 'TOGETHER/ALONE']  F: ['CAN/CANNOT', 'MUST/LET']
```

### Full Pipeline with Governance Trace

```bash
python demo.py
```

```
  PHASE 1: PROPOSE — Crystallizing structure from noise...
  PHASE 2: DECIDE  — 7/7 admissibility gates passed
  PHASE 3: PROMOTE — 7/7 witnesses attested, commitment locked
  PHASE 4: EXECUTE — Generating within committed envelope...

    module prim_secded_64_57_bind_fpv;

      bind prim_secded_64_57_tb
        prim_secded_64_57_assert_fpv prim_secded_64_57_assert_fpv (
        .clk_i,
        .rst_ni,
        .data_i,
        .data_o,

  GOVERNANCE TRACE
    Gates passed:      7/7
    Witnesses:         7/7 unanimous
    Phases completed:  PROPOSE -> DECIDE -> PROMOTE -> EXECUTE
```

Saves a machine-verifiable `governance_trace.json` with every output.

### Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MetaCortex-Dynamics/Axiom/blob/main/Axiom_Demo.ipynb)

### Retrain from Scratch

```bash
mkdir -p corpus/public
git clone --depth 1 https://github.com/lowRISC/ibex.git corpus/public/ibex
git clone --depth 1 https://github.com/lowRISC/opentitan.git corpus/public/opentitan
git clone --depth 1 https://github.com/chipsalliance/Cores-SweRV.git corpus/public/Cores-SweRV

python scripts/build_decoder_v2.py
```

## Results

| Variant | Final Loss | Converges? |
|---------|-----------|-----------|
| A (hierarchical) | **0.107** | **Yes** |
| B (flat) | 0.181 | Yes |
| C (uniform) | 0.110 | Yes |
| D (inverted) | 0.094 | **No** (memorization) |

A beats B by 41%. D achieves lowest loss but never converges structurally — the convergence diagnostic correctly identifies memorization.

100% pipeline admission rate. Decoder generates valid SystemVerilog from governed structures.

## License

MIT. See [LICENSE](LICENSE).

## Organization

[MetaCortex Dynamics DAO](https://github.com/MetaCortex-Dynamics)
