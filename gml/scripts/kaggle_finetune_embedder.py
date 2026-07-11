"""Kaggle-optimized FT-3: fine-tune bge-large-en-v1.5 embedder on LOCOMO QA pairs.

Paste this whole file into one Kaggle notebook cell and hit Run All.

Kaggle setup before running:
  - Notebook → Settings → Internet: ON
  - Accelerator: GPU T4 x1   (P100 also works)
  - No Kaggle dataset upload needed

What this does, vs FT-2 (Qwen LoRA):
  FT-2 made the *generator* terse. FT-3 makes the *embedder* better at
  putting LOCOMO-relevant evidence near the question vector in embedding
  space, so retrieval surfaces the right memories more often. Bigger
  win for cat-2 multi-hop where retrieval recall is the bottleneck.

Method:
  sentence-transformers MultipleNegativesRankingLoss on (question, evidence)
  positive pairs. In-batch negatives are mined automatically — no explicit
  negative sampling step needed. ~335M params, full bf16/fp16 fine-tune
  (NOT LoRA — embedder is small enough to tune end-to-end).

Output:
  /kaggle/working/embedder_locomo_ft/        ← SentenceTransformer model dir
  /kaggle/working/embedder_locomo_ft.zip     ← zipped for one-click download
  /kaggle/working/embedder_locomo_ft/ft_summary.json  ← pre/post cosine-sim metrics
"""
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path


# ---- Config -----------------------------------------------------------------
BASE_MODEL = "BAAI/bge-large-en-v1.5"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_out"
OUT_DIR = WORK_DIR / "embedder_locomo_ft"
DATA_DIR = WORK_DIR / "locomo_emb_data"

# Same 8-train / 2-test conversation split as scripts/prepare_ft_data.py
# (holds out conv-49 and conv-50 for measurement).
TRAIN_CONV_IDS = set(range(8))

# Training hyperparams. bge-large is 335M params — comfortably fits a T4 at
# batch=16 in fp16, no QLoRA needed. Larger batch = more in-batch negatives
# per anchor = stronger contrastive signal.
EPOCHS = 2
BATCH = 16
LR = 2e-5
WARMUP_RATIO = 0.1

PIP_INSTALL = "sentence-transformers>=2.7 datasets>=2.18"


# ---- 0. Install deps --------------------------------------------------------
print("[0/5] installing dependencies (silent)...")
os.system(f"pip install -q --upgrade {PIP_INSTALL}")


# ---- 1. Download LOCOMO + build (q, evidence) pairs ------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

locomo_path = DATA_DIR / "locomo10.json"
if not locomo_path.exists():
    print(f"[1/5] downloading LOCOMO from {LOCOMO_URL}")
    urllib.request.urlretrieve(LOCOMO_URL, locomo_path)
print(f"  LOCOMO: {locomo_path.stat().st_size / 1e6:.1f} MB")


def _index_conversation(conv: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    block = conv.get("conversation", conv)
    if not isinstance(block, dict):
        return out
    for key, messages in block.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(messages, list):
            continue
        for m in messages:
            if not isinstance(m, dict):
                continue
            did = m.get("dia_id")
            text = m.get("text") or m.get("content") or ""
            if did and text:
                speaker = m.get("speaker") or ""
                out[did] = f"{speaker}: {text}" if speaker else text
    return out


print("[2/5] building (question, evidence) pairs...")
locomo = json.loads(locomo_path.read_text(encoding="utf-8"))
train_pairs: list[tuple[str, str]] = []
test_pairs: list[tuple[str, str]] = []
seen: set[tuple[str, str]] = set()

for i, conv in enumerate(locomo):
    conv_idx = _index_conversation(conv)
    target = train_pairs if i in TRAIN_CONV_IDS else test_pairs
    for qa in conv.get("qa", []):
        q = qa.get("question", "").strip()
        if not q:
            continue
        evidence = qa.get("evidence") or []
        if not isinstance(evidence, list):
            continue
        for eid in evidence:
            if isinstance(eid, str) and eid in conv_idx:
                positive = conv_idx[eid]
                key = (q, positive)
                # Dedup across the whole split; same (q, p) repeats are wasted
                # gradient steps without adding signal.
                if key in seen:
                    continue
                seen.add(key)
                target.append(key)

print(f"  {len(train_pairs)} train / {len(test_pairs)} test unique (q, positive) pairs")


# ---- 2. Load base model ----------------------------------------------------
print(f"[3/5] loading base: {BASE_MODEL}")
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
print(f"  device: {device} ({gpu_name})")

t0 = time.time()
model = SentenceTransformer(BASE_MODEL, device=device)
print(f"  loaded in {time.time() - t0:.1f}s")


# ---- 3. Pre-train cosine sim baseline (50 test pairs) ----------------------
sample = test_pairs[:50] if len(test_pairs) >= 50 else test_pairs

def _avg_cos_sim(pairs):
    if not pairs:
        return float("nan")
    q_vec = model.encode([q for q, _ in pairs], convert_to_numpy=True, show_progress_bar=False)
    p_vec = model.encode([p for _, p in pairs], convert_to_numpy=True, show_progress_bar=False)
    return float(np.mean(
        (q_vec * p_vec).sum(axis=1)
        / (np.linalg.norm(q_vec, axis=1) * np.linalg.norm(p_vec, axis=1) + 1e-9)
    ))

pre_sim = _avg_cos_sim(sample)
print(f"  pre-train avg cosine sim (q, positive) on {len(sample)} test pairs: {pre_sim:.3f}")


# ---- 4. Train --------------------------------------------------------------
print(f"[4/5] training: epochs={EPOCHS} batch={BATCH} lr={LR}")
train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = int(len(train_loader) * EPOCHS * WARMUP_RATIO)
print(f"  steps_per_epoch={len(train_loader)} warmup={warmup_steps}")

t0 = time.time()
model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
)
train_min = (time.time() - t0) / 60
print(f"  trained in {train_min:.1f} min")


# ---- 5. Post-train cosine sim, save, zip -----------------------------------
post_sim = _avg_cos_sim(sample)
print(f"  post-train avg cosine sim: {post_sim:.3f}  (Δ {post_sim - pre_sim:+.3f})")

print(f"[5/5] saving to {OUT_DIR}")
model.save(str(OUT_DIR))

summary = {
    "base_model": BASE_MODEL,
    "device": device,
    "gpu": gpu_name,
    "epochs": EPOCHS,
    "batch_size": BATCH,
    "lr": LR,
    "warmup_steps": warmup_steps,
    "train_pairs": len(train_pairs),
    "test_pairs": len(test_pairs),
    "training_minutes": round(train_min, 1),
    "pre_avg_cos_sim": round(pre_sim, 4),
    "post_avg_cos_sim": round(post_sim, 4),
    "delta_cos_sim": round(post_sim - pre_sim, 4),
}
(OUT_DIR / "ft_summary.json").write_text(json.dumps(summary, indent=2))

zip_path = WORK_DIR / "embedder_locomo_ft.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT_DIR.rglob("*"):
        if p.is_file():
            z.write(p, arcname=p.relative_to(OUT_DIR.parent))
print(f"  zipped → {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

print("\nDone. Download from the Kaggle sidebar:")
print("  - embedder_locomo_ft.zip       (the fine-tuned SentenceTransformer)")
print("  - embedder_locomo_ft/ft_summary.json")
print("\nIntegration on your local machine:")
print("  1. unzip into models/embedder_locomo_ft/")
print("  2. Set GML_EMBEDDER_MODEL=models/embedder_locomo_ft to use it")
print("  (or merge into the bench config; see docs/gpu-setup.md)")
