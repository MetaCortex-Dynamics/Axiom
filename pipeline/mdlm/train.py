"""
MDLM Training Loop — Train on governed corpus, measure convergence.

Usage:
  python -m pipeline.mdlm.train --corpus corpus/CORPUS-FINAL --schedule A --epochs 50

Convergence metric: per-epoch accuracy on masked token prediction.
PHI_CONTRACTION_RATE threshold: loss ratio between consecutive epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Install with: pip install torch")

from pipeline.mdlm.tokenizer import load_corpus, pad_sequence, VOCAB_SIZE, decode
from pipeline.mdlm.model import (
    MaskingSchedule, StructureModel, compute_loss, generate,
)

PHI_CONTRACTION_RATE = 0.381966  # (3 - sqrt(5)) / 2


def train(
    corpus_dir: str,
    schedule: MaskingSchedule,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    total_timesteps: int = 100,
    max_len: int = 40,
    device_name: str = "cpu",
):
    """Train MDLM on governed corpus."""
    print(f"=== MDLM Training ===")
    print(f"Schedule: {schedule.value} ({schedule.name})")
    print(f"Corpus: {corpus_dir}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print()

    # Load and encode corpus
    sequences = load_corpus(corpus_dir)
    print(f"Loaded {len(sequences)} sequences")

    # Pad to fixed length
    padded = [pad_sequence(seq, max_len) for seq in sequences]
    data = torch.tensor(padded, dtype=torch.long)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    device = torch.device(device_name)
    model = StructureModel(
        vocab_size=VOCAB_SIZE, d_model=128, nhead=4,
        num_layers=4, max_len=max_len,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    history = []
    prev_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            # Random timestep per batch
            timestep = random.randint(0, total_timesteps)

            loss = compute_loss(model, batch, schedule, timestep, total_timesteps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        avg_loss = epoch_loss / max(epoch_batches, 1)

        # Convergence ratio: loss_t / loss_{t-1}
        ratio = avg_loss / prev_loss if prev_loss > 0 and prev_loss != float("inf") else 1.0
        prev_loss = avg_loss

        # Regime classification
        if ratio < PHI_CONTRACTION_RATE:
            regime = "CONTRACTING"
        elif ratio < 0.8:
            regime = "BOUNDARY"
        elif ratio < 1.0:
            regime = "CRITICAL"
        else:
            regime = "DIVERGENT"

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "ratio": ratio,
            "regime": regime,
        })

        if epoch <= 5 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  ratio={ratio:.4f}  [{regime}]")

    print()

    # Generate samples
    print("Generating 5 samples...")
    model.eval()
    samples = generate(model, 5, max_len, schedule, total_timesteps)
    for i in range(5):
        seq = samples[i].tolist()
        print(f"  Sample {i}: {decode(seq)}")
    print()

    # Summary
    final = history[-1]
    contracting_epochs = sum(1 for h in history if h["regime"] == "CONTRACTING")
    print(f"=== SUMMARY ===")
    print(f"Final loss: {final['loss']:.4f}")
    print(f"Final ratio: {final['ratio']:.4f} ({final['regime']})")
    print(f"Contracting epochs: {contracting_epochs}/{len(history)}")
    print(f"Schedule: {schedule.value} ({schedule.name})")

    return history


def main():
    if not HAS_TORCH:
        sys.exit(1)

    parser = argparse.ArgumentParser(description="MDLM Training")
    parser.add_argument("--corpus", required=True, help="Corpus directory")
    parser.add_argument("--schedule", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Save history to JSON")
    args = parser.parse_args()

    schedule = MaskingSchedule(args.schedule)
    history = train(
        args.corpus, schedule, args.epochs, args.batch_size,
        args.lr, device_name=args.device,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved to {args.output}")


if __name__ == "__main__":
    main()
