"""
Distillation prompts — convert committed governance structures to governed prose.

Each committed structure becomes a prompt to a frontier LLM.
The LLM generates candidate prose. The pipeline filters it.
T-status prose joins the training corpus.
"""

from __future__ import annotations

from pipeline.mdlm.tokenizer import TOKEN_NAMES, OP_OFFSET


SYSTEM_PROMPT = """You are a technical writer producing governed prose for semiconductor verification.

Given a governance structure with three channels (G, S, F) containing operators,
produce a single coherent passage of technical prose that expresses the structure.

Rules:
- Output must be valid technical content (SystemVerilog, specifications, or documentation)
- Every claim must be traceable to the operators provided
- Do not introduce claims not supported by the operators
- Do not use persuasive, coercive, or authority-asserting language
- Be precise, factual, and technical"""


def build_distill_prompt(gov_structure: dict) -> str:
    """Build a distillation prompt from a governance structure.

    Args:
        gov_structure: dict with keys 'G', 'S', 'F', each containing
                      a list of operator dicts with 'operator' field.

    Returns:
        Prompt string for the LLM.
    """
    g_ops = [op["operator"] for op in gov_structure.get("G", [])]
    s_ops = [op["operator"] for op in gov_structure.get("S", [])]
    f_ops = [op["operator"] for op in gov_structure.get("F", [])]

    return f"""Express the following governance structure as technical semiconductor prose.

Channel G (provenance): {', '.join(g_ops)}
Channel S (structure): {', '.join(s_ops)}
Channel F (function): {', '.join(f_ops)}

Requirements:
- The prose must be a valid SystemVerilog module, specification section, or technical document fragment
- Channel G operators establish what this content IS and where it comes from
- Channel S operators establish the structural relationships
- Channel F operators establish what it DOES and what constraints it satisfies
- Output only the technical prose, no explanation or meta-commentary

Prose:"""


def build_synthetic_structures(n: int = 1000) -> list[dict]:
    """Generate synthetic governance structures for distillation.

    Creates compositional variations of operator assignments
    across the three channels, respecting home conditions.
    """
    import random
    random.seed(42)

    # Home operators per channel (opaque tier names)
    tier_1 = ["THIS", "SAME/NOT-SAME", "NO"]
    tier_2 = ["GOES-WITH", "TOGETHER/ALONE", "MANY/ONE", "EVERY/SOME", "MORE/LESS", "CAN/CANNOT"]
    tier_3 = ["INSIDE/OUTSIDE", "NEAR/FAR", "IF/THEN", "BECAUSE", "MAYBE", "MUST/LET"]

    structures = []
    for _ in range(n):
        # G channel: 1-3 operators, must include at least one tier_1
        g_count = random.randint(1, 3)
        g_ops = [random.choice(tier_1)]
        for _ in range(g_count - 1):
            g_ops.append(random.choice(tier_1 + tier_3[:2]))  # cross-modal allowed

        # S channel: 2-5 operators, must include at least one tier_2
        s_count = random.randint(2, 5)
        s_ops = [random.choice(tier_2)]
        for _ in range(s_count - 1):
            s_ops.append(random.choice(tier_2 + tier_3[:2]))

        # F channel: 2-4 operators, must include at least one tier_3
        f_count = random.randint(2, 4)
        f_ops = [random.choice(tier_3)]
        for _ in range(f_count - 1):
            f_ops.append(random.choice(tier_3))

        structures.append({
            "G": [{"operator": op} for op in g_ops],
            "S": [{"operator": op} for op in s_ops],
            "F": [{"operator": op} for op in f_ops],
        })

    return structures
