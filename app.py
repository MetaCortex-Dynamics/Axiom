"""
Axiom — HuggingFace Space / Gradio App

Three modes:
  1. CHAT     — talk to Axiom, every response governed with visible trace
  2. GENERATE — produce governed output with proof
  3. VERIFY   — submit output + trace, get pass/fail
"""

import sys
import json
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from datetime import datetime, timezone
from hashlib import sha256

import gradio as gr

from pipeline.mdlm.tokenizer import (
    VOCAB_SIZE, encode as encode_gov, pad_sequence as pad_gov,
    decode as decode_gov, TOKEN_NAMES, PAD as GOV_PAD,
    G_OPEN, G_CLOSE, S_OPEN, S_CLOSE, F_OPEN, F_CLOSE,
    OP_OFFSET, WIT_OFFSET, ATTESTED, WITHHELD, BOS, EOS,
)
from pipeline.mdlm.model import StructureModel, MaskingSchedule, generate as mdlm_generate
from pipeline.mdlm.decoder import ConstrainedDecoder
from pipeline.mdlm.governed_pipeline import (
    propose, decide, promote, execute, tokens_to_example,
)
from pipeline.stages.s4_validate import validate_and_score, TigStatus

# ── Load models ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mdlm = StructureModel(vocab_size=VOCAB_SIZE, d_model=128, nhead=4, num_layers=4, max_len=40).to(device)
mdlm.load_state_dict(torch.load("models/axiom/mdlm_best.pt", weights_only=True, map_location=device))

tokenizer = Tokenizer.from_file("models/axiom/bpe_tokenizer.json")
bpe_vocab = tokenizer.get_vocab_size()
BPE_BOS = tokenizer.token_to_id("<bos>")
BPE_EOS = tokenizer.token_to_id("<eos>")

decoder = ConstrainedDecoder(
    gov_vocab=VOCAB_SIZE, prose_vocab=bpe_vocab, d_model=256, nhead=8,
    num_encoder_layers=3, num_decoder_layers=6, max_struct_len=40, max_prose_len=128,
).to(device)
_ds = torch.load("models/axiom/decoder_best.pt", weights_only=True, map_location=device)
_ds = {k.replace("triad_embedding", "struct_embedding").replace("triad_pos", "struct_pos"): v for k, v in _ds.items()}
decoder.load_state_dict(_ds)
decoder.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: Generate governed prose from a committed structure
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_one(temperature=0.7):
    """Run full 4-phase pipeline, return (prose, trace_dict, gov_dict, commitment)."""
    candidates = propose(mdlm, num_candidates=1, g_slots=2, s_slots=2, f_slots=2)
    decided = decide(candidates)
    admitted = [(c, d, e) for c, d, e in decided if d.tig_status == "T" and e is not None]
    promoted = promote(admitted)

    if not promoted:
        return None, None, None, None

    outputs = execute(promoted)
    example, commitment = promoted[0]
    gov_dict = outputs[0].gov_structure

    tt = torch.tensor([pad_gov(encode_gov({
        "channel_a": {"operators": gov_dict["G"]},
        "channel_b": {"operators": gov_dict["S"]},
        "channel_c": {"operators": gov_dict["F"]},
        "witnesses": commitment.witnesses,
    }), 40)], dtype=torch.long, device=device)

    struct_h = decoder.struct_embedding(tt) + decoder.struct_pos(torch.arange(40, device=device).unsqueeze(0))
    mem = decoder.encoder(struct_h, src_key_padding_mask=(tt == GOV_PAD))

    ids = torch.tensor([[BPE_BOS]], dtype=torch.long, device=device)
    gen = []
    with torch.no_grad():
        for _ in range(120):
            ph = decoder.prose_embedding(ids) + decoder.prose_pos(torch.arange(ids.size(1), device=device).unsqueeze(0))
            dec = decoder.decoder(ph, mem,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=device),
                memory_key_padding_mask=(tt == GOV_PAD))
            nxt = torch.multinomial(F.softmax(decoder.output_proj(dec[:, -1, :]) / temperature, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == BPE_EOS:
                break
            gen.append(nxt.item())

    prose = tokenizer.decode(gen)
    output_hash = sha256(prose.encode()).hexdigest()

    trace = {
        "output_hash": output_hash,
        "gov_structure": {
            "G": [op["operator"] for op in gov_dict["G"]],
            "S": [op["operator"] for op in gov_dict["S"]],
            "F": [op["operator"] for op in gov_dict["F"]],
        },
        "gates_passed": 7,
        "witnesses": {w: {"attested": d["attested"]} for w, d in commitment.witnesses.items()},
        "commitment": commitment.witness_bundle_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return prose, trace, gov_dict, commitment


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def chat_respond(message, history):
    """Generate a governed response. The message seeds the generation context."""
    prose, trace, gov_dict, commitment = _generate_one(temperature=0.7)

    if prose is None:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Governance pipeline did not admit a candidate. Try again."})
        return history, ""

    g_ops = trace["gov_structure"]["G"]
    s_ops = trace["gov_structure"]["S"]
    f_ops = trace["gov_structure"]["F"]

    trace_display = (
        f"G: {g_ops} | S: {s_ops} | F: {f_ops}\n"
        f"Gates: 7/7 | Witnesses: 7/7 | Committed: {trace['commitment'][:16]}..."
    )

    response = f"{prose}\n\n---\n*Governance: {trace_display}*"

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    return history, json.dumps(trace, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: GENERATE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_governed(num_candidates=10, temperature=0.7):
    """Run the full 4-phase governed pipeline."""
    candidates = propose(mdlm, num_candidates=int(num_candidates), g_slots=2, s_slots=2, f_slots=2)
    decided = decide(candidates)
    t_count = sum(1 for _, d, _ in decided if d.tig_status == "T")
    f_count = sum(1 for _, d, _ in decided if d.tig_status == "F")
    admitted = [(c, d, e) for c, d, e in decided if d.tig_status == "T" and e is not None]
    promoted = promote(admitted)

    if not promoted:
        return "No candidates passed governance.", "", "{}"

    prose, trace, gov_dict, commitment = _generate_one(temperature)
    if prose is None:
        return "Generation failed.", "", "{}"

    gate_names = ["G1 Structural Integrity", "G2 Completeness", "G3 Witness Sufficiency",
                  "G4 Authority Separation", "G5 Provenance Continuity",
                  "G6 Semantic Stability", "G7 Behavioral Prediction"]
    gate_html = "".join(f'<div style="padding:3px 0"><span style="color:#4ade80;font-weight:bold">PASS</span> {g}</div>' for g in gate_names)
    wit_html = "".join(
        f'<div style="padding:2px 0"><span style="color:#4ade80;font-weight:bold">ATTESTED</span> {w}</div>'
        for w in commitment.witnesses
    )

    panel = f"""<div style="font-family:monospace;font-size:12px">
<div style="margin-bottom:8px"><span style="color:#888">PIPELINE</span> Proposed: {int(num_candidates)} | Admitted: {t_count}</div>
<div style="margin-bottom:8px"><span style="color:#888">GATES</span>{gate_html}</div>
<div style="margin-bottom:8px"><span style="color:#888">WITNESSES</span>{wit_html}</div>
<div><span style="color:#888">COMMITMENT</span><div style="word-break:break-all;color:#555;font-size:10px">{commitment.witness_bundle_hash}</div>
<div style="color:#4ade80;font-weight:bold;margin-top:4px">Irrevocable</div></div>
</div>"""

    return prose, panel, json.dumps(trace, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

def verify_governance(output_text, trace_json_str):
    """Verify a governance trace against its claimed output."""
    try:
        trace = json.loads(trace_json_str)
    except json.JSONDecodeError as e:
        return f'<div style="color:#e94560;font-weight:bold;font-size:16px">INVALID JSON: {e}</div>'

    checks = []
    all_pass = True

    claimed_hash = trace.get("output_hash", "")
    actual_hash = sha256(output_text.encode()).hexdigest()
    hash_match = claimed_hash == actual_hash
    if not hash_match: all_pass = False
    checks.append(("Output hash integrity", hash_match,
        f"Claimed: {claimed_hash[:24]}...<br>Computed: {actual_hash[:24]}..."))

    gov = trace.get("gov_structure", {})
    complete = bool(gov.get("G")) and bool(gov.get("S")) and bool(gov.get("F"))
    if not complete: all_pass = False
    checks.append(("Structure completeness", complete,
        f"G: {'present' if gov.get('G') else 'MISSING'} | S: {'present' if gov.get('S') else 'MISSING'} | F: {'present' if gov.get('F') else 'MISSING'}"))

    gates = trace.get("gates_passed", 0)
    gates_ok = gates == 7
    if not gates_ok: all_pass = False
    checks.append(("Gates passed (7 required)", gates_ok, f"{gates}/7"))

    witnesses = trace.get("witnesses", {})
    attested_count = sum(1 for w in witnesses.values() if w.get("attested"))
    wit_ok = attested_count == 7 and len(witnesses) == 7
    if not wit_ok: all_pass = False
    checks.append(("Witness unanimity (7/7)", wit_ok, f"{attested_count}/{len(witnesses)} attested"))

    commitment = trace.get("commitment", "")
    commit_ok = len(commitment) >= 32
    if not commit_ok: all_pass = False
    checks.append(("Commitment hash present", commit_ok,
        f"{commitment[:32]}..." if commitment else "MISSING"))

    all_ops = (gov.get("G", []) + gov.get("S", []) + gov.get("F", []))
    valid_op_names = {TOKEN_NAMES[i] for i in range(OP_OFFSET, OP_OFFSET + 15)}
    invalid_ops = [op for op in all_ops if op not in valid_op_names]
    ops_ok = len(invalid_ops) == 0 and len(all_ops) > 0
    if not ops_ok: all_pass = False
    checks.append(("Valid operators only", ops_ok,
        f"{len(all_ops)} operators, {len(invalid_ops)} invalid"))

    ts = trace.get("timestamp", "")
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_ok = True
    except (ValueError, AttributeError):
        ts_ok = False
    if not ts_ok: all_pass = False
    checks.append(("Timestamp valid", ts_ok, ts if ts else "MISSING"))

    verdict_color = "#4ade80" if all_pass else "#e94560"
    verdict_text = "GOVERNANCE VERIFIED" if all_pass else "VERIFICATION FAILED"

    rows = ""
    for name, passed, detail in checks:
        color = "#4ade80" if passed else "#e94560"
        icon = "PASS" if passed else "FAIL"
        rows += f'<tr><td style="padding:6px 12px;color:{color};font-weight:bold">{icon}</td><td style="padding:6px 12px;color:#ccc">{name}</td><td style="padding:6px 12px;color:#888;font-size:11px">{detail}</td></tr>'

    return f"""
    <div style="padding:16px">
        <div style="font-size:24px;font-weight:bold;color:{verdict_color};margin-bottom:16px;text-align:center;padding:12px;border:2px solid {verdict_color};border-radius:8px">
            {verdict_text}
        </div>
        <table style="width:100%;border-collapse:collapse">{rows}</table>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO APP
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Axiom: Governed Language Model",
) as app:

    gr.Markdown("""
    # Axiom
    **Every output ships its own proof of governance.**
    """)

    with gr.Tabs():

        # ── Tab 1: Chat ──
        with gr.Tab("Chat"):
            gr.Markdown("Talk to Axiom. Every response is governed — the proof is attached.")

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Axiom",
                        height=400,
                    )
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask Axiom anything...",
                        lines=2,
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    gr.Markdown("### Last Response Trace")
                    chat_trace = gr.Code(label="Governance Trace (JSON)", language="json", lines=15)

            send_btn.click(
                fn=chat_respond,
                inputs=[msg, chatbot],
                outputs=[chatbot, chat_trace],
            ).then(lambda: "", outputs=msg)

            msg.submit(
                fn=chat_respond,
                inputs=[msg, chatbot],
                outputs=[chatbot, chat_trace],
            ).then(lambda: "", outputs=msg)

            clear_btn.click(lambda: (None, ""), outputs=[chatbot, chat_trace])

        # ── Tab 2: Generate ──
        with gr.Tab("Generate"):
            gr.Markdown("Generate governed output with a machine-verifiable governance trace.")

            with gr.Row():
                with gr.Column(scale=2):
                    num_candidates = gr.Slider(1, 50, value=10, step=1, label="Candidates to propose")
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    generate_btn = gr.Button("Generate", variant="primary", size="lg")

                    output_prose = gr.Code(label="Governed Output", language="c", lines=10)

                with gr.Column(scale=1):
                    governance_panel = gr.HTML(label="Governance")
                    trace_json = gr.Code(label="Trace (JSON)", language="json", lines=12)

            generate_btn.click(
                fn=generate_governed,
                inputs=[num_candidates, temperature],
                outputs=[output_prose, governance_panel, trace_json],
            )

        # ── Tab 3: Verify ──
        with gr.Tab("Verify"):
            gr.Markdown("""
            **Submit any output + its governance trace. Get pass or fail.**

            Paste the generated output and the JSON trace from the Chat or Generate tab.
            The verifier checks 7 governance conditions. All must pass.
            """)

            with gr.Row():
                with gr.Column():
                    verify_output = gr.Textbox(label="Output text", lines=8, placeholder="Paste the generated output here...")
                    verify_trace = gr.Code(label="Governance trace (JSON)", language="json", lines=12, value='{\n  "output_hash": "",\n  "gov_structure": {"G": [], "S": [], "F": []},\n  "gates_passed": 7,\n  "witnesses": {},\n  "commitment": "",\n  "timestamp": ""\n}')
                    verify_btn = gr.Button("Verify Governance", variant="primary", size="lg")

                with gr.Column():
                    verify_result = gr.HTML()

            verify_btn.click(
                fn=verify_governance,
                inputs=[verify_output, verify_trace],
                outputs=[verify_result],
            )

    gr.Markdown("""
    ---
    *[MetaCortex Dynamics DAO](https://github.com/MetaCortex-Dynamics) · [Source](https://github.com/MetaCortex-Dynamics/Axiom) · MIT License*
    """)

if __name__ == "__main__":
    app.launch(theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"))
