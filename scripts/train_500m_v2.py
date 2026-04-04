"""
PLAN-AXIOM-001 P2: Train 500M on 32K pairs (3 teachers).
All 9 review fixes applied. Resumable. Drive-safe. Logged.
"""

import sys, os, json, random, time, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer
from pipeline.mdlm.tokenizer import encode as encode_gov, pad_sequence as pad_gov, VOCAB_SIZE as SV, PAD as SP
from pipeline.mdlm.decoder_500m import Axiom500MDecoder, count_params

DEVICE = "cuda"
MAX_S = 40; MAX_P = 256; BATCH = 8; ACCUM = 4; LR = 2e-4; EPOCHS = 50; WARM = 3

# Paths
LOCAL_DIR = "models/axiom-500m-v2"
LOCAL_CKPT = f"{LOCAL_DIR}/checkpoint.pt"
DRIVE_DIR = "/content/drive/MyDrive"
DRIVE_CKPT = f"{DRIVE_DIR}/axiom_500m_32k_checkpoint.pt"
DRIVE_TMP = f"{DRIVE_DIR}/axiom_500m_32k_checkpoint.tmp.pt"
DRIVE_LOG = f"{DRIVE_DIR}/axiom_500m_train.log"


def log(msg, log_file=None):
    """[MUST] Print and persist every message."""
    print(msg, flush=True)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except:
            pass


def load_pairs():
    pairs = []
    for path in ["corpus/axiom/pairs.json", "corpus/distilled/self_distilled_pairs.json", "corpus/teacher/teacher_pairs.json"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for p in data:
                pairs.append({"frame": p.get("frame", p.get("triad", {})), "prose": p["prose"]})
            log(f"  {path}: {len(data)}", DRIVE_LOG if os.path.exists(DRIVE_DIR) else None)
    log(f"  Total: {len(pairs)}", DRIVE_LOG if os.path.exists(DRIVE_DIR) else None)
    return pairs


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best, path):
    """[MUST] Save full resumable checkpoint."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
    }
    torch.save(ckpt, path)
    return ckpt


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """[CAN] Resume from checkpoint if it exists."""
    if not os.path.exists(path):
        return 0, 999.0
    try:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        log(f"  [RESUME] Loaded checkpoint from epoch {ckpt['epoch']}, best={ckpt['best']:.4f}")
        return ckpt["epoch"], ckpt["best"]
    except Exception as e:
        log(f"  [RESUME] Failed to load checkpoint: {e}. Starting fresh.")
        return 0, 999.0


def main():
    has_drive = os.path.exists(DRIVE_DIR)
    lf = DRIVE_LOG if has_drive else None

    log("=== P2: Train 500M on 32K pairs (3 teachers) ===", lf)

    # [MUST] Create checkpoint directories
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # [MUST] Verify Drive space if mounted
    if has_drive:
        usage = shutil.disk_usage(DRIVE_DIR)
        free_gb = usage.free / 1e9
        log(f"  Drive free: {free_gb:.1f}GB", lf)
        assert free_gb > 5, f"FAIL: only {free_gb:.1f}GB free on Drive, need >5GB"
    else:
        log("  WARNING: Drive not mounted. Saving locally only.", lf)

    tok = Tokenizer.from_file("models/axiom/bpe_tokenizer.json")
    bv = tok.get_vocab_size()
    PAD_T = tok.token_to_id("<pad>"); BOS = tok.token_to_id("<bos>"); EOS = tok.token_to_id("<eos>")

    pairs = load_pairs()
    sd, pi, pt = [], [], []
    for p in pairs:
        try:
            st = pad_gov(encode_gov(p["frame"]), MAX_S)
        except:
            continue
        bpe = [BOS] + tok.encode(p["prose"]).ids[:MAX_P-2] + [EOS]
        while len(bpe) < MAX_P: bpe.append(PAD_T)
        bpe = bpe[:MAX_P]
        sd.append(st); pi.append(bpe[:-1]); pt.append(bpe[1:])

    log(f"  Encoded: {len(sd)}", lf)
    loader = DataLoader(
        TensorDataset(torch.tensor(sd, dtype=torch.long), torch.tensor(pi, dtype=torch.long), torch.tensor(pt, dtype=torch.long)),
        batch_size=BATCH, shuffle=True, drop_last=True,
    )

    dev = torch.device(DEVICE)
    model = Axiom500MDecoder(
        struct_vocab=SV, prose_vocab=bv, d_model=1024, nhead=16,
        num_encoder_layers=6, num_decoder_layers=24,
        max_struct_len=MAX_S, max_prose_len=MAX_P, use_checkpoint=True,
    ).to(dev)
    log(f"  {count_params(model)/1e6:.0f}M params, batch {BATCH}x{ACCUM}={BATCH*ACCUM}", lf)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler("cuda")
    total_steps = EPOCHS * len(loader)
    warm_steps = WARM * len(loader)
    sched = optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm_steps, 1) if s < warm_steps
        else 0.5 * (1 + torch.cos(torch.tensor((s - warm_steps) / max(total_steps - warm_steps, 1) * 3.14159)).item())
    )

    # [CAN] Resume from Drive checkpoint if available
    start_epoch = 0
    best = 999.0
    if has_drive and os.path.exists(DRIVE_CKPT):
        start_epoch, best = load_checkpoint(DRIVE_CKPT, model, opt, sched, scaler, dev)
    elif os.path.exists(LOCAL_CKPT):
        start_epoch, best = load_checkpoint(LOCAL_CKPT, model, opt, sched, scaler, dev)

    t0 = time.time()

    for ep in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        el = 0; b = 0
        opt.zero_grad()

        for i, (sb, ib, tb) in enumerate(loader):
            sb, ib, tb = sb.to(dev), ib.to(dev), tb.to(dev)
            with autocast("cuda", dtype=torch.bfloat16):
                losses = model.compute_protocol_loss(sb, ib, tb, prose_pad_id=PAD_T)
                loss = losses["total"] / ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            el += losses["total"].item() if hasattr(losses["total"], "item") else losses["total"]
            b += 1

        avg = el / max(b, 1)

        if avg < best:
            best = avg

            # [MUST] Save full resumable checkpoint locally
            save_checkpoint(model, opt, sched, scaler, ep, best, LOCAL_CKPT)

            # [MUST] Save to Drive with atomic rename
            if has_drive:
                ckpt = {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best": best,
                }
                torch.save(ckpt, DRIVE_TMP)
                os.replace(DRIVE_TMP, DRIVE_CKPT)

            log(f"  [SAVE] epoch={ep} best={best:.6f} local={LOCAL_CKPT} drive={'YES' if has_drive else 'NO'}", lf)

        if ep <= 3 or ep % 5 == 0 or ep == EPOCHS:
            elapsed = time.time() - t0
            lr = opt.param_groups[0]["lr"]
            log(f"E{ep:3d}: L={avg:.4f} ce={losses['ce']:.4f} best={best:.4f} lr={lr:.6f} {elapsed:.0f}s", lf)

    elapsed = time.time() - t0
    log(f"\nDone: {elapsed:.0f}s ({elapsed/max(EPOCHS-start_epoch,1):.1f}s/epoch), best={best:.4f}", lf)

    # [MUST] Final model-only save for inference (smaller file)
    model_only_path = f"{LOCAL_DIR}/decoder_best.pt"
    if has_drive and os.path.exists(DRIVE_CKPT):
        ckpt = torch.load(DRIVE_CKPT, map_location=dev)
        model.load_state_dict(ckpt["model"])
    torch.save(model.state_dict(), model_only_path)
    if has_drive:
        torch.save(model.state_dict(), f"{DRIVE_DIR}/axiom_500m_decoder_final.pt")
        log(f"  [SAVE] Final model-only saved to Drive", lf)

    # Generation samples
    model.eval()
    log("\n--- Generation ---", lf)
    for i in range(5):
        p = pairs[i]
        try:
            st = torch.tensor([pad_gov(encode_gov(p["frame"]), MAX_S)], dtype=torch.long, device=dev)
            gen = model.generate(st, BOS, EOS, max_len=150, temperature=0.7)
            log(f"\n[{i}] Src: {p['prose'][:80]}\n    Gen: {tok.decode(gen[0])[:80]}", lf)
        except Exception as e:
            log(f"\n[{i}] Error: {e}", lf)

    log("\n=== P2 retrain complete. Evaluate G4. ===", lf)


if __name__ == "__main__":
    main()
