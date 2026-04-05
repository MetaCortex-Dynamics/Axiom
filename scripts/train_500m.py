"""
Train Axiom 500M decoder.

Protocol loss: L = L_ce + 0.1*L_telos + 0.1*L_witness + 0.05*L_audit
Gradient checkpointing + bf16 for 8GB VRAM.
"""

import sys
import os
import json
import random
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer

from pipeline.mdlm.tokenizer import (
    encode as encode_gov, pad_sequence as pad_gov,
    VOCAB_SIZE as STRUCT_VOCAB, TOKEN_NAMES, PAD as STRUCT_PAD,
)
from pipeline.mdlm.decoder_500m import Axiom500MDecoder, count_params

DEVICE = "cuda"
MAX_STRUCT_LEN = 40
MAX_PROSE_LEN = 256
BATCH_SIZE = 4        # small for 8GB VRAM
GRAD_ACCUM = 8        # effective batch = 32
LR = 3e-4
EPOCHS = 100
WARMUP = 5


def main():
    print("=== Train Axiom 500M ===")

    # Load BPE tokenizer
    tokenizer = Tokenizer.from_file("models/axiom/bpe_tokenizer.json")
    bpe_vocab = tokenizer.get_vocab_size()
    BPE_PAD = tokenizer.token_to_id("<pad>")
    BPE_BOS = tokenizer.token_to_id("<bos>")
    BPE_EOS = tokenizer.token_to_id("<eos>")

    # Load pairs
    with open("corpus/axiom/pairs.json", encoding="utf-8") as f:
        raw_pairs = json.load(f)
    print(f"Pairs: {len(raw_pairs)}, BPE vocab: {bpe_vocab}")

    # Encode
    struct_data = []
    prose_input = []
    prose_target = []

    for p in raw_pairs:
        st = pad_gov(encode_gov(p["frame"] if "frame" in p else p.get("triad", p)), MAX_STRUCT_LEN)
        bpe = [BPE_BOS] + tokenizer.encode(p["prose"]).ids[:MAX_PROSE_LEN - 2] + [BPE_EOS]
        while len(bpe) < MAX_PROSE_LEN:
            bpe.append(BPE_PAD)
        bpe = bpe[:MAX_PROSE_LEN]
        struct_data.append(st)
        prose_input.append(bpe[:-1])
        prose_target.append(bpe[1:])

    loader = DataLoader(
        TensorDataset(
            torch.tensor(struct_data, dtype=torch.long),
            torch.tensor(prose_input, dtype=torch.long),
            torch.tensor(prose_target, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    # Model
    device = torch.device(DEVICE)
    model = Axiom500MDecoder(
        struct_vocab=STRUCT_VOCAB,
        prose_vocab=bpe_vocab,
        d_model=1024, nhead=16,
        num_encoder_layers=6,
        num_decoder_layers=24,
        max_struct_len=MAX_STRUCT_LEN,
        max_prose_len=MAX_PROSE_LEN,
        use_checkpoint=True,
    ).to(device)

    params = count_params(model)
    print(f"Model: {params:,} parameters ({params/1e6:.0f}M)")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"VRAM budget: gradient checkpointing + bf16")
    print()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler("cuda")

    # Warmup + cosine schedule
    def lr_lambda(step):
        if step < WARMUP * len(loader):
            return step / max(WARMUP * len(loader), 1)
        progress = (step - WARMUP * len(loader)) / max((EPOCHS - WARMUP) * len(loader), 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs("models/axiom-500m", exist_ok=True)
    t0 = time.time()
    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = {"total": 0, "ce": 0, "telos": 0, "witness": 0, "audit": 0}
        batches = 0
        optimizer.zero_grad()

        for i, (sb, ib, tb) in enumerate(loader):
            sb, ib, tb = sb.to(device), ib.to(device), tb.to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                losses = model.compute_protocol_loss(
                    sb, ib, tb, prose_pad_id=BPE_PAD,
                    alpha_telos=0.1, alpha_witness=0.1, alpha_audit=0.05,
                )
                loss = losses["total"] / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if k != "total" else losses[k].item() * GRAD_ACCUM

            batches += 1

        avg = {k: v / max(batches, 1) for k, v in epoch_losses.items()}

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save(model.state_dict(), "models/axiom-500m/decoder_best.pt")

        if epoch <= 3 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"E{epoch:3d}: L={avg['total']:.4f} ce={avg['ce']:.4f} "
                f"tel={avg['telos']:.4f} wit={avg['witness']:.4f} aud={avg['audit']:.4f} "
                f"lr={lr:.6f} best={best_loss:.4f} {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.0f}s ({elapsed/EPOCHS:.1f}s/epoch), best={best_loss:.4f}")

    # Generate samples
    print("\n--- Generation ---")
    model.load_state_dict(torch.load("models/axiom-500m/decoder_best.pt", weights_only=True))
    model.eval()

    for i in range(3):
        p = raw_pairs[i]
        st = torch.tensor(
            [pad_gov(encode_gov(p["frame"] if "frame" in p else p.get("triad", p)), MAX_STRUCT_LEN)],
            dtype=torch.long, device=device,
        )
        gen_ids = model.generate(st, BPE_BOS, BPE_EOS, max_len=150, temperature=0.7)
        gen_text = tokenizer.decode(gen_ids[0])
        print(f"\n[{i}] Source: {p['prose'][:80]}")
        print(f"    Gen:    {gen_text[:80]}")

    print("\n=== Phase 1 complete. Evaluate G3. ===")


if __name__ == "__main__":
    main()
