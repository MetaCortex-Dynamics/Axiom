"""
hierarchical Hypothesis Test — 4-way comparison per PLAN-GHA-002 §4.4.

Variant A: hierarchical hierarchical (Tier 1 → Tier 2 → CL+readiness)
Variant B: flat hierarchical (operators only)
Variant C: Uniform random masking
Variant D: inverted inverted

Prediction: A > B > C > D on convergence rate.
Key comparison: A vs B isolates readiness dimensions.
"""

import sys
import json
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from pipeline.mdlm.train import train
from pipeline.mdlm.model import MaskingSchedule

CORPUS = "corpus/CORPUS-FINAL"
EPOCHS = 50
DEVICE = "cuda"

results = {}

for schedule in [MaskingSchedule.HIERARCHICAL, MaskingSchedule.FLAT,
                 MaskingSchedule.UNIFORM, MaskingSchedule.INVERTED]:
    print(f"\n{'='*60}")
    print(f"VARIANT {schedule.value}: {schedule.name}")
    print(f"{'='*60}\n")

    history = train(CORPUS, schedule, epochs=EPOCHS, device_name=DEVICE)
    results[schedule.value] = history

    with open(f"results_variant_{schedule.value}.json", "w") as f:
        json.dump(history, f, indent=2)

# Summary comparison
print(f"\n{'='*60}")
print("HYPOTHESIS TEST RESULTS")
print(f"{'='*60}\n")
print(f"{'Variant':<12} {'Final Loss':>12} {'Min Ratio':>12} {'Contracting':>12} {'First Contract':>15}")
print("-" * 65)

for variant in ["A", "B", "C", "D"]:
    h = results[variant]
    final_loss = h[-1]["loss"]
    min_ratio = min(e["ratio"] for e in h[1:])  # skip first
    contracting = sum(1 for e in h if e["regime"] == "CONTRACTING")
    first_contract = next((e["epoch"] for e in h if e["regime"] == "CONTRACTING"), "never")
    print(f"{variant:<12} {final_loss:>12.4f} {min_ratio:>12.4f} {contracting:>12} {str(first_contract):>15}")

print()
# Verdict
a_loss = results["A"][-1]["loss"]
b_loss = results["B"][-1]["loss"]
c_loss = results["C"][-1]["loss"]
d_loss = results["D"][-1]["loss"]
print(f"A vs B (readiness): {'A wins' if a_loss < b_loss else 'B wins'} ({a_loss:.4f} vs {b_loss:.4f})")
print(f"A vs C (hierarchy vs uniform): {'A wins' if a_loss < c_loss else 'C wins'} ({a_loss:.4f} vs {c_loss:.4f})")
print(f"C vs D (uniform vs inverted): {'C wins' if c_loss < d_loss else 'D wins'} ({c_loss:.4f} vs {d_loss:.4f})")
print(f"Predicted order A > B > C > D: {a_loss < b_loss < c_loss < d_loss}")
