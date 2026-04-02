"""
Axiom — Simple Demo

Generate governed semiconductor code in one call.
"""

import sys
sys.path.insert(0, ".")

import torch
from pipeline.mdlm.tokenizer import VOCAB_SIZE
from pipeline.mdlm.model import StructureModel
from pipeline.mdlm.governed_pipeline import run_governed_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = StructureModel(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, max_len=40).to(device)
model.load_state_dict(torch.load("models/axiom/mdlm_best.pt", weights_only=True, map_location=device))

# Generate
report = run_governed_pipeline(model, num_candidates=10, g_slots=2, s_slots=2, f_slots=2)

print(f"{report.executed}/{report.proposed} outputs passed all governance gates\n")
for o in report.outputs[:3]:
    g = [x["operator"] for x in o["gov_structure"]["G"]]
    s = [x["operator"] for x in o["gov_structure"]["S"]]
    f = [x["operator"] for x in o["gov_structure"]["F"]]
    print(f"  G: {g}  S: {s}  F: {f}")
