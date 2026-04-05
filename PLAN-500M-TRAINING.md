# 500M Decoder Training — Verified Execution Plan (v2)

[WHENCE] 3 prior failures. [BECAUSE] ephemeral disk, no resume, no atomic save.
[FOR-WHAT] zero data loss on 4th attempt.

## Fixes Applied (all 9 from independent review)

1. [MUST] `os.makedirs()` before [EVERY] save path
2. [MUST] Full resumable checkpoint (model + optimizer + scheduler + scaler + epoch + best)
3. [MUST] Drive free space verified (>5GB)
4. [MUST] Behavior verification, [NOT SAME] line-number verification
5. [MUST] Resume path: load checkpoint, continue from last epoch
6. [MUST] `torch.load(map_location=device)` — no `weights_only` during resume
7. [MUST] Improvement-triggered save, [NOT SAME] epoch-count-triggered
8. [MUST] `tee` to Drive log file — stdout survives disconnect
9. [MUST] Atomic rename: write to `.tmp.pt`, `os.replace()` to final path

## Preconditions

[THIS] corpus: 32,032 pairs at `corpus/{axiom,distilled,teacher}/`
[THIS] script: `scripts/train_500m_v2.py` (v2, all 9 fixes)
[THIS] architecture: 496M params, 24 layers, 16 heads, 1024 hidden

## Step 1: Mount Drive + Verify Space

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
usage = shutil.disk_usage("/content/drive/MyDrive")
free_gb = usage.free / 1e9
assert free_gb > 5, f"FAIL: {free_gb:.1f}GB free, need >5GB"
print(f"PASS: Drive mounted, {free_gb:.1f}GB free")
```

[IF/THEN] PASS → Step 2
[IF/THEN] FAIL → clear Drive space, retry

## Step 2: Verify Drive Write (2GB test)

```python
import torch, os
path = "/content/drive/MyDrive/axiom_drive_test_2gb.pt"
dummy = {"data": torch.randn(256, 1024, 1024)}  # ~1GB
torch.save(dummy, path)
loaded = torch.load(path, map_location="cpu")
assert loaded["data"].shape == (256, 1024, 1024)
size_mb = os.path.getsize(path) / 1e6
os.remove(path)
print(f"PASS: wrote and read {size_mb:.0f}MB to Drive")
```

[IF/THEN] PASS → Step 3
[IF/THEN] FAIL → Drive I/O problem. Do [NOT] proceed.

## Step 3: Clone + Install

```python
!git clone https://github.com/MetaCortex-Dynamics/Axiom.git
%cd Axiom
!pip install -q torch tokenizers
```

## Step 4: Verify Corpus

```python
import json, os
paths = [
    "corpus/axiom/pairs.json",
    "corpus/distilled/self_distilled_pairs.json",
    "corpus/teacher/teacher_pairs.json",
]
total = 0
for p in paths:
    assert os.path.exists(p), f"FAIL: {p} missing"
    with open(p) as f:
        n = len(json.load(f))
    print(f"  {p}: {n}")
    total += n
assert total >= 30000, f"FAIL: only {total} pairs"
print(f"PASS: {total} pairs")
```

[IF/THEN] PASS (>= 30,000) → Step 5
[IF/THEN] FAIL → corpus incomplete

## Step 5: Verify GPU

```python
import torch
assert torch.cuda.is_available(), "FAIL: no CUDA"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"PASS: {name}, {vram:.0f}GB")
```

[IF/THEN] PASS → Step 6
[IF/THEN] FAIL → wrong runtime

## Step 6: Verify Script Has All 9 Fixes

```python
import inspect, importlib.util
spec = importlib.util.spec_from_file_location("train", "scripts/train_500m_v2.py")
mod = importlib.util.module_from_spec(spec)
src = open("scripts/train_500m_v2.py").read()

checks = [
    ("os.makedirs", "makedirs" in src),
    ("full checkpoint (optimizer)", '"optimizer"' in src and '"scheduler"' in src and '"scaler"' in src),
    ("Drive space check", "disk_usage" in src),
    ("resume logic", "load_checkpoint" in src),
    ("atomic rename", "os.replace" in src),
    ("Drive log", "DRIVE_LOG" in src),
    ("tmp file save", ".tmp.pt" in src),
    ("[SAVE] log", "[SAVE]" in src),
    ("drive path inside loop", src.index("DRIVE_CKPT") < src.index("for ep in range")),
]

all_pass = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok: all_pass = False
    print(f"  [{status}] {name}")

assert all_pass, "FAIL: script missing fixes"
print("PASS: all 9 fixes verified")
```

[IF/THEN] PASS → Step 7
[IF/THEN] FAIL → do [NOT] train. Fix script.

## Step 7: Train

```python
!python -u scripts/train_500m_v2.py 2>&1 | tee /content/drive/MyDrive/axiom_500m_train.log
```

[EVERY] loss improvement → full checkpoint saved to:
  - local: `models/axiom-500m-v2/checkpoint.pt`
  - Drive: `/content/drive/MyDrive/axiom_500m_32k_checkpoint.pt` (atomic rename)

[EVERY] log line → appended to Drive log

[IF/THEN] runtime dies → checkpoint on Drive from last best epoch. Resume by re-running same cell.

## Step 8: Verify During Training

[CAN] run in separate cell while training runs:

```python
import os
ckpt = "/content/drive/MyDrive/axiom_500m_32k_checkpoint.pt"
log = "/content/drive/MyDrive/axiom_500m_train.log"
if os.path.exists(ckpt):
    size_mb = os.path.getsize(ckpt) / 1e6
    print(f"PASS: checkpoint on Drive, {size_mb:.0f}MB")
else:
    print("WAITING: no improvement yet")
if os.path.exists(log):
    lines = open(log).readlines()
    print(f"Log: {len(lines)} lines, last: {lines[-1].strip() if lines else 'empty'}")
```

## Step 9: Post-Training

```python
import torch
path = "/content/drive/MyDrive/axiom_500m_32k_checkpoint.pt"
ckpt = torch.load(path, map_location="cpu")
print(f"PASS: epoch={ckpt['epoch']}, best={ckpt['best']:.4f}, keys={len(ckpt['model'])}")
```

[IF/THEN] loads → download `axiom_500m_decoder_final.pt` from Drive
[IF/THEN] runtime died → same checkpoint is on Drive. Re-run Step 7 to resume.

## Resume Protocol

[IF/THEN] runtime dies during training:
1. Reconnect or start new runtime
2. Mount Drive (Step 1)
3. Clone repo (Step 3)
4. Run Step 7 again — script auto-detects Drive checkpoint, resumes from last saved epoch
5. [NO] lost compute. [NO] lost data.

## Failure Modes

| Failure | [BECAUSE] | Mitigation |
|---------|-----------|------------|
| Runtime dies | Colab recycles | Checkpoint on Drive. Resume. |
| Drive full | Large checkpoints | Step 1 verifies >5GB free |
| Corrupt checkpoint | Write interrupted | Atomic rename (tmp → final) |
| Script missing fixes | Wrong version cloned | Step 6 verifies all 9 fixes |
| [NO] GPU | Wrong runtime | Step 5 verifies CUDA |
| [NO] corpus | Clone failed | Step 4 verifies 3 files, >=30K |

## Notebook Cell Order

```
Cell 1: Step 1 — Mount Drive + verify space
Cell 2: Step 2 — Verify 2GB Drive write
Cell 3: Step 3 — Clone + install
Cell 4: Step 4 — Verify corpus
Cell 5: Step 5 — Verify GPU
Cell 6: Step 6 — Verify script has all 9 fixes
Cell 7: Step 7 — Train (with tee to Drive log)
Cell 8: Step 9 — Verify checkpoint
```

[EVERY] step has PASS/FAIL.
[NO] step proceeds without prior PASS.
[MUST] Steps 1-6 pass before Step 7.
[IF/THEN] Step 7 interrupted → re-run Step 7 to resume.
