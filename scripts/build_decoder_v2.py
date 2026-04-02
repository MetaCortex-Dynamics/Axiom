"""
Phase 5 v2: BPE decoder — Steps 1-3 in one run.

Step 1: Pair governed examples with source prose
Step 2: Train BPE tokenizer on the prose corpus
Step 3: Train encoder-decoder with BPE tokenization
"""

import sys
import os
import json
import random
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from pipeline.mdlm.tokenizer import (
    encode as encode_gov, pad_sequence as pad_gov,
    VOCAB_SIZE as STRUCT_VOCAB, TOKEN_NAMES, PAD as GOV_PAD,
)
from pipeline.ingest.chat_archive import ingest_conversation_file
from pipeline.stages.s2_classify import classify, Classification
from pipeline.stages.s3_decompose import decompose
from pipeline.stages.s4_validate import validate_and_score, TigStatus
from pipeline.mdlm.decoder import ConstrainedDecoder

# For the public reference model, use pre-extracted pairs from the public corpus.
# To retrain on your own corpus, set this to your conversation directory.
THEORY_DIR = None  # Set to path of conversation JSONs, or None to use corpus/axiom/pairs.json
MAX_STRUCT_LEN = 40
MAX_PROSE_LEN = 128  # BPE tokens, not chars
BPE_VOCAB_SIZE = 8192
EPOCHS = 100
BATCH_SIZE = 32
LR = 5e-4
DEVICE = "cuda"


def step1_extract_pairs():
    """Step 1: Extract (structure tokens, source prose) pairs."""
    print("=== Step 1: Extract structure-prose pairs ===")
    from pathlib import Path
    t0 = time.time()

    pairs = []  # (struct_dict, prose_text, source_id)
    theory_path = Path(THEORY_DIR)

    for conv_file in sorted(theory_path.glob("conv_*.json")):
        try:
            for seg in ingest_conversation_file(conv_file):
                c = classify(seg)
                if c.classification != Classification.TECHNICAL:
                    continue
                ex = decompose(c)
                if ex is None:
                    continue
                r = validate_and_score(ex)
                if r.tig_status != TigStatus.TRUE:
                    continue

                struct_dict = {
                    "channel_a": {"operators": [
                        {"operator": e.operator.canonical_name, "evidence": e.evidence}
                        for e in ex.channel_a.operators.expressions
                    ]},
                    "channel_b": {"operators": [
                        {"operator": e.operator.canonical_name, "evidence": e.evidence}
                        for e in ex.channel_b.operators.expressions
                    ]},
                    "channel_c": {"operators": [
                        {"operator": e.operator.canonical_name, "evidence": e.evidence}
                        for e in ex.channel_c.operators.expressions
                    ]},
                    "witnesses": {
                        w.canonical_name: {"attested": a.attested, "evidence": a.evidence}
                        for w, a in ex.witnesses.attestations.items()
                    },
                }
                pairs.append((struct_dict, seg.text, ex.provenance.source_id))
        except Exception:
            continue

    print(f"  Extracted {len(pairs)} pairs in {time.time()-t0:.0f}s")
    return pairs


def step2_train_bpe(pairs):
    """Step 2: Train BPE tokenizer on the prose side."""
    print("\n=== Step 2: Train BPE tokenizer ===")
    t0 = time.time()

    # Write prose to temp file for tokenizer training
    prose_file = "models/prose_corpus.txt"
    os.makedirs("models", exist_ok=True)
    with open(prose_file, "w", encoding="utf-8") as f:
        for _, prose, _ in pairs:
            f.write(prose + "\n")

    # Train BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
    )
    tokenizer.train([prose_file], trainer)
    tokenizer.save("models/bpe_tokenizer.json")

    print(f"  BPE vocab: {tokenizer.get_vocab_size()} tokens")
    print(f"  Trained in {time.time()-t0:.0f}s")

    # Test
    sample = pairs[0][1][:100]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Sample: '{sample[:60]}...'")
    print(f"  Tokens: {len(encoded.ids)}")
    print(f"  Roundtrip: '{decoded[:60]}...'")

    return tokenizer


def step3_train_decoder(pairs, tokenizer):
    """Step 3: Train encoder-decoder with BPE."""
    print(f"\n=== Step 3: Train BPE decoder ({EPOCHS} epochs) ===")

    BPE_PAD = tokenizer.token_to_id("<pad>")
    BPE_BOS = tokenizer.token_to_id("<bos>")
    BPE_EOS = tokenizer.token_to_id("<eos>")
    actual_vocab = tokenizer.get_vocab_size()

    # Encode all pairs
    struct_data = []
    prose_input = []
    prose_target = []

    for struct_dict, prose, _ in pairs:
        gov_tokens = encode_gov(struct_dict)
        struct_padded = pad_gov(gov_tokens, MAX_STRUCT_LEN)

        bpe_ids = tokenizer.encode(prose).ids
        # Prepend BOS, append EOS
        bpe_seq = [BPE_BOS] + bpe_ids[:MAX_PROSE_LEN - 2] + [BPE_EOS]
        # Pad
        while len(bpe_seq) < MAX_PROSE_LEN:
            bpe_seq.append(BPE_PAD)
        bpe_seq = bpe_seq[:MAX_PROSE_LEN]

        # Teacher forcing: input = [BOS, t1, t2, ...], target = [t1, t2, ..., EOS]
        struct_data.append(struct_padded)
        prose_input.append(bpe_seq[:-1])
        prose_target.append(bpe_seq[1:])

    # Cap training size
    max_train = min(len(struct_data), 5000)
    if len(struct_data) > max_train:
        random.seed(42)
        indices = random.sample(range(len(struct_data)), max_train)
        struct_data = [struct_data[i] for i in indices]
        prose_input = [prose_input[i] for i in indices]
        prose_target = [prose_target[i] for i in indices]

    struct_tensor = torch.tensor(struct_data, dtype=torch.long)
    input_tensor = torch.tensor(prose_input, dtype=torch.long)
    target_tensor = torch.tensor(prose_target, dtype=torch.long)

    print(f"  Training pairs: {len(struct_data)}")
    print(f"  BPE vocab: {actual_vocab}")

    dataset = TensorDataset(struct_tensor, input_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    device = torch.device(DEVICE)
    model = ConstrainedDecoder(
        gov_vocab=STRUCT_VOCAB,
        prose_vocab=actual_vocab,
        d_model=256, nhead=8,
        num_encoder_layers=3, num_decoder_layers=6,
        max_struct_len=MAX_STRUCT_LEN,
        max_prose_len=MAX_PROSE_LEN,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {param_count:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    t0 = time.time()
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for struct_batch, inp_batch, tgt_batch in loader:
            struct_batch = struct_batch.to(device)
            inp_batch = inp_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            logits = model(struct_batch, inp_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, actual_vocab),
                tgt_batch.reshape(-1),
                ignore_index=BPE_PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/ggp_decoder_bpe_best.pt")

        if epoch <= 3 or epoch % 10 == 0 or epoch == EPOCHS:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  E{epoch:3d}: loss={avg_loss:.4f}  best={best_loss:.4f}  lr={lr:.6f}  {elapsed:.0f}s", flush=True)

    print(f"\n  Training done: {time.time()-t0:.0f}s, best loss={best_loss:.4f}")

    # Generate samples
    print("\n=== Generation samples ===")
    model.load_state_dict(torch.load("models/ggp_decoder_bpe_best.pt", weights_only=True))
    model.eval()

    for i in range(5):
        struct_dict, original_prose, source_id = pairs[i]
        gov_tokens = encode_gov(struct_dict)
        struct_padded = pad_gov(gov_tokens, MAX_STRUCT_LEN)
        struct_tensor = torch.tensor([struct_padded], dtype=torch.long, device=device)

        # Generate
        generated_ids = []
        input_ids = torch.tensor([[BPE_BOS]], dtype=torch.long, device=device)

        with torch.no_grad():
            # Encode governed
            struct_len = struct_tensor.size(1)
            struct_pos = torch.arange(struct_len, device=device).unsqueeze(0)
            struct_h = model.struct_embedding(struct_tensor) + model.struct_pos(struct_pos)
            struct_pad_mask = (struct_tensor == GOV_PAD)
            memory = model.encoder(struct_h, src_key_padding_mask=struct_pad_mask)

            for _ in range(100):
                prose_len = input_ids.size(1)
                prose_pos = torch.arange(prose_len, device=device).unsqueeze(0)
                prose_h = model.prose_embedding(input_ids) + model.prose_pos(prose_pos)
                causal = nn.Transformer.generate_square_subsequent_mask(prose_len, device=device)
                decoded = model.decoder(prose_h, memory, tgt_mask=causal, memory_key_padding_mask=struct_pad_mask)
                logits = model.output_proj(decoded[:, -1, :]) / 0.8
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
                if next_tok.item() == BPE_EOS:
                    break
                generated_ids.append(next_tok.item())

        generated_text = tokenizer.decode(generated_ids)
        ops_g = [e.operator.canonical_name for e in pairs[i][0].get("channel_a", {}).get("operators", [])] if isinstance(pairs[i][0], dict) else "?"

        gov_ops = [TOKEN_NAMES[t] for t in struct_padded if 10 <= t <= 24]
        print(f"\n  [{i}] governed ops: {gov_ops}")
        print(f"      Source:    {original_prose[:120]}...")
        print(f"      Generated: {generated_text[:120]}...")

    return model


def main():
    # Step 1
    pairs = step1_extract_pairs()
    if len(pairs) < 100:
        print("Too few pairs. Aborting.")
        return

    # Step 2
    tokenizer = step2_train_bpe(pairs)

    # Step 3
    model = step3_train_decoder(pairs, tokenizer)

    print("\n=== COMPLETE ===")
    print(f"Pairs: {len(pairs)}")
    print(f"BPE vocab: {tokenizer.get_vocab_size()}")
    print(f"Models saved: models/bpe_tokenizer.json, models/ggp_decoder_bpe_best.pt")


if __name__ == "__main__":
    main()
