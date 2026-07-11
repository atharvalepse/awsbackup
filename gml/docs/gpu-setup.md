# Switching the FT workload to a CUDA GPU (RTX 4050 / 4060 / 4070)

This is the minimum to pick up where the Mac stopped.

## 1. Prereqs

- NVIDIA driver + CUDA 12.x toolkit (`nvidia-smi` should work)
- Python 3.11 or 3.12
- ~12 GB free disk (model + caches)

## 2. Sync the repo

```bash
git clone <this-repo> gml-orchestration
cd gml-orchestration
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ft,ft-cuda]"
```

`ft-cuda` pulls `bitsandbytes` for 4-bit QLoRA — required if your GPU
has ≤ 6 GB VRAM (RTX 4050). Skip the `ft-cuda` extra on Windows; install
`bitsandbytes-windows-webui` separately if needed.

## 3. Confirm CUDA is visible to torch

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True  NVIDIA GeForce RTX 4050 Laptop GPU
```

## 4. Move the data files

Copy these from the old Mac (or regenerate with the prepare-* scripts):

```
/tmp/locomo/locomo10.json              # raw LOCOMO benchmark
/tmp/locomo-ft-data/train.jsonl        # CE training triples
/tmp/locomo-ft-data/test.jsonl
/tmp/locomo-llm-ft-data/train.jsonl    # LLM (prompt, completion) pairs
/tmp/locomo-llm-ft-data/test.jsonl
models/ce_locomo_ft/                   # FT'd cross-encoder (FT-1)
```

If `/tmp/locomo*` aren't around, regenerate:
```bash
.venv/bin/python scripts/prepare_ft_data.py --data /tmp/locomo/locomo10.json --out /tmp/locomo-ft-data
.venv/bin/python scripts/prepare_ft_llm_data.py --data /tmp/locomo/locomo10.json --out /tmp/locomo-llm-ft-data
```

## 5. Run the chain

`run_ft_chain.sh` auto-detects CUDA and enables 4-bit QLoRA automatically.

```bash
# Full chain (mining → CE retrain → LLM FT → eval → embedder FT)
bash scripts/run_ft_chain.sh full

# Or just the LLM side (FT-2 + eval)
bash scripts/run_ft_chain.sh llm-only

# Or just the embedder
bash scripts/run_ft_chain.sh ft-emb
```

Overrides:
- `GML_FT_DEVICE=cuda|mps|cpu` to force a device
- `GML_FT_4BIT=0` to disable 4-bit even on CUDA (for ≥ 12 GB GPUs)
- `GML_EVAL_MAX=394` to run LLM eval on the full test set (default 100)

## 6. Expected wall-clock on RTX 4050 (6 GB)

| Step | RTX 4050 (4-bit) | M2 Max MPS |
|---|---|---|
| FT-1 CE retrain on hard negatives | ~3 min | ~12 min |
| FT-2 Qwen2.5-3B LoRA (1 epoch, 1583 ex) | ~15 min | ~30+ min |
| FT-3 bge-large embedder (1 epoch) | ~4 min | ~20 min |
| LLM eval (100 ex, base+FT) | ~6 min | ~15 min |
| Hard-negative mining | ~3 min | ~10 min |

QLoRA on 4-bit makes a measurable accuracy difference vs full bf16 LoRA
(~0.5-1 F1 point). If you can grab a 12 GB+ machine later, drop
`GML_FT_4BIT=0` and re-run FT-2 for the cleanest numbers.

## 7. Wire FT'd LLM back into orchestration

After FT-2 finishes, point the runtime at the adapter:

```bash
export GML_LLM_BACKEND=transformers
export GML_TRANSFORMERS_BASE=Qwen/Qwen2.5-3B-Instruct
export GML_TRANSFORMERS_ADAPTER=$(pwd)/models/qwen_locomo_ft/adapter
export GML_TRANSFORMERS_DEVICE=cuda
.venv/bin/python scripts/benchmark_locomo.py --data /tmp/locomo/locomo10.json --convs 8,9
```

This is hot-swappable — no rebuild needed.
