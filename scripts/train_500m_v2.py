"""
PLAN-AXIOM-001 P2: Retrain 500M on expanded corpus (12K pairs).
"""

import sys, os, json, random, time
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

def load_pairs():
    pairs = []
    for path in ["corpus/axiom/pairs.json", "corpus/distilled/self_distilled_pairs.json"]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for p in data:
                pairs.append({"frame": p.get("frame", p.get("triad", {})), "prose": p["prose"]})
            print(f"  {path}: {len(data)}")
    print(f"  Total: {len(pairs)}")
    return pairs

def main():
    print("=== P2: Retrain 500M on 12K pairs ===")
    tok = Tokenizer.from_file("models/axiom/bpe_tokenizer.json")
    bv = tok.get_vocab_size()
    PAD_T = tok.token_to_id("<pad>"); BOS = tok.token_to_id("<bos>"); EOS = tok.token_to_id("<eos>")

    pairs = load_pairs()
    sd, pi, pt = [], [], []
    for p in pairs:
        try: st = pad_gov(encode_gov(p["frame"]), MAX_S)
        except: continue
        bpe = [BOS] + tok.encode(p["prose"]).ids[:MAX_P-2] + [EOS]
        while len(bpe) < MAX_P: bpe.append(PAD_T)
        bpe = bpe[:MAX_P]
        sd.append(st); pi.append(bpe[:-1]); pt.append(bpe[1:])

    print(f"  Encoded: {len(sd)}")
    loader = DataLoader(TensorDataset(torch.tensor(sd,dtype=torch.long),torch.tensor(pi,dtype=torch.long),torch.tensor(pt,dtype=torch.long)),batch_size=BATCH,shuffle=True,drop_last=True)

    dev = torch.device(DEVICE)
    model = Axiom500MDecoder(struct_vocab=SV,prose_vocab=bv,d_model=1024,nhead=16,num_encoder_layers=6,num_decoder_layers=24,max_struct_len=MAX_S,max_prose_len=MAX_P,use_checkpoint=True).to(dev)
    print(f"  {count_params(model)/1e6:.0f}M params, batch {BATCH}x{ACCUM}={BATCH*ACCUM}")

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9,0.95))
    scaler = GradScaler("cuda")
    total_steps = EPOCHS * len(loader)
    warm_steps = WARM * len(loader)
    sched = optim.lr_scheduler.LambdaLR(opt, lambda s: s/max(warm_steps,1) if s<warm_steps else 0.5*(1+torch.cos(torch.tensor((s-warm_steps)/(total_steps-warm_steps)*3.14159)).item()))

    os.makedirs("models/axiom-500m-v2", exist_ok=True)
    t0 = time.time(); best = 999.0

    for ep in range(1, EPOCHS+1):
        model.train(); el = 0; b = 0; opt.zero_grad()
        for i, (sb,ib,tb) in enumerate(loader):
            sb,ib,tb = sb.to(dev),ib.to(dev),tb.to(dev)
            with autocast("cuda", dtype=torch.bfloat16):
                losses = model.compute_protocol_loss(sb,ib,tb,prose_pad_id=PAD_T)
                loss = losses["total"] / ACCUM
            scaler.scale(loss).backward()
            if (i+1)%ACCUM==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            el += losses["total"].item() if hasattr(losses["total"],"item") else losses["total"]
            b += 1
        avg = el/max(b,1)
        if avg < best: best = avg; torch.save(model.state_dict(),"models/axiom-500m-v2/decoder_best.pt")
        if ep<=3 or ep%5==0 or ep==EPOCHS:
            print(f"E{ep:3d}: L={avg:.4f} ce={losses['ce']:.4f} best={best:.4f} {time.time()-t0:.0f}s",flush=True)

    print(f"\nDone: {time.time()-t0:.0f}s best={best:.4f}")

    model.load_state_dict(torch.load("models/axiom-500m-v2/decoder_best.pt",weights_only=True)); model.eval()
    for i in range(5):
        p = pairs[i]
        try:
            st = torch.tensor([pad_gov(encode_gov(p["frame"]),MAX_S)],dtype=torch.long,device=dev)
            gen = model.generate(st,BOS,EOS,max_len=150,temperature=0.7)
            print(f"\n[{i}] Src: {p['prose'][:80]}\n    Gen: {tok.decode(gen[0])[:80]}")
        except Exception as e: print(f"\n[{i}] Error: {e}")

    print("\n=== P2 retrain complete. Evaluate G4. ===")

if __name__ == "__main__":
    main()
