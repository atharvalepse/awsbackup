#!/bin/bash
# Chain the three fine-tunes + eval between each.
# Each stage frees MPS memory before launching the next so we don't OOM.
#
# Usage:
#   bash scripts/run_ft_chain.sh [stage]
# where stage ∈ {eval-ce, ft-llm, ft-emb, full}.
# Defaults to "full" which runs every step in sequence.

set -eu
cd "$(dirname "$0")/.."   # repo root, regardless of where it's run from

DATA="${GML_LOCOMO_DATA:-/tmp/locomo/locomo10.json}"
CE_DIR=models/ce_locomo_ft
CE_HN_DIR=models/ce_locomo_ft_hn        # CE retrained with hard negatives
LLM_DIR=models/qwen_locomo_ft
EMB_DIR=models/embedder_locomo_ft
HN_DATA=/tmp/locomo-ft-data-hard

# Detect compute backend. GML_FT_DEVICE overrides (cuda|mps|cpu).
DEVICE="${GML_FT_DEVICE:-}"
if [ -z "$DEVICE" ]; then
  if .venv/bin/python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
  elif .venv/bin/python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE=mps
  else
    DEVICE=cpu
  fi
fi
echo "[run-ft-chain] device: $DEVICE"

# QLoRA on CUDA = 6 GB GPUs can fit Qwen2.5-3B. Default ON for CUDA.
USE_4BIT_FLAG=""
if [ "$DEVICE" = "cuda" ] && [ "${GML_FT_4BIT:-1}" = "1" ]; then
  USE_4BIT_FLAG="--use-4bit"
fi

free_mps() {
  .venv/bin/python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
" 2>/dev/null || true
}

step_eval_ce() {
  echo "=========================================="
  echo "  Step: retrieval eval (base CE vs FT'd CE)"
  echo "=========================================="
  if [ ! -d "$CE_DIR" ]; then
    echo "  [skip] $CE_DIR not present — run FT-1 first"
    return 1
  fi
  free_mps
  .venv/bin/python scripts/eval_retrieval.py \
    --data "$DATA" \
    --test-conv-indices 8,9 \
    --base-ce BAAI/bge-reranker-base \
    --cand-ce "$CE_DIR" \
    --k-values 5,10,20 \
    --max-qa-per-conv 30 2>&1 | tail -25
}

step_ft_llm() {
  echo "=========================================="
  echo "  Step: FT-2  Qwen2.5-3B LoRA fine-tune"
  echo "=========================================="
  free_mps
  # On CUDA: bf16 + 4-bit base = fits in 6 GB (RTX 4050) and trains ~5x
  # faster than MPS. On MPS: full-precision base + adamw_torch.
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 \
  TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  .venv/bin/python scripts/finetune_answer_llm.py \
    --data-dir /tmp/locomo-llm-ft-data \
    --base Qwen/Qwen2.5-3B-Instruct \
    --out "$LLM_DIR" \
    --epochs 1 --batch 1 --grad-accum 4 \
    --max-len 1024 --lora-r 16 \
    $USE_4BIT_FLAG 2>&1 | tail -40
}

step_eval_llm() {
  echo "=========================================="
  echo "  Step: LLM answer-quality eval (base vs FT'd)"
  echo "=========================================="
  if [ ! -d "$LLM_DIR/adapter" ]; then
    echo "  [skip] $LLM_DIR/adapter not present — run ft-llm first"
    return 1
  fi
  free_mps
  .venv/bin/python scripts/eval_llm.py \
    --base Qwen/Qwen2.5-3B-Instruct \
    --adapter "$LLM_DIR/adapter" \
    --data /tmp/locomo-llm-ft-data/test.jsonl \
    --device "$DEVICE" \
    --max-examples "${GML_EVAL_MAX:-100}" 2>&1 | tail -30
}

step_ft_emb() {
  echo "=========================================="
  echo "  Step: FT-3  bge-large-en-v1.5 fine-tune"
  echo "=========================================="
  free_mps
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python scripts/finetune_embedder.py \
    --data-dir /tmp/locomo-ft-data \
    --base BAAI/bge-large-en-v1.5 \
    --out "$EMB_DIR" \
    --epochs 1 --batch 8 2>&1 | tail -30
}

step_mine_hn() {
  echo "=========================================="
  echo "  Step: hard-negative mining (uses FT'd CE)"
  echo "=========================================="
  if [ ! -d "$CE_DIR" ]; then
    echo "  [skip] $CE_DIR not present — run FT-1 first"
    return 1
  fi
  free_mps
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python scripts/mine_hard_negatives.py \
    --ce-model "$CE_DIR" \
    --data "$DATA" \
    --in-data /tmp/locomo-ft-data \
    --out-dir "$HN_DATA" \
    --negs-per-positive 4 \
    --candidates-per-query 50 2>&1 | tail -20
}

step_ft_ce_hn() {
  echo "=========================================="
  echo "  Step: CE retrain on hard negatives"
  echo "=========================================="
  if [ ! -d "$HN_DATA" ]; then
    echo "  [skip] $HN_DATA not present — run mine-hn first"
    return 1
  fi
  free_mps
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  .venv/bin/python scripts/finetune_cross_encoder.py \
    --data-dir "$HN_DATA" \
    --base "$CE_DIR" \
    --out "$CE_HN_DIR" \
    --epochs 1 --batch 4 --lr 1e-5 2>&1 | tail -30
}

step_eval_ce_hn() {
  echo "=========================================="
  echo "  Step: retrieval eval (FT'd CE vs HN-retrained CE)"
  echo "=========================================="
  if [ ! -d "$CE_HN_DIR" ]; then
    echo "  [skip] $CE_HN_DIR not present — run ft-ce-hn first"
    return 1
  fi
  free_mps
  .venv/bin/python scripts/eval_retrieval.py \
    --data "$DATA" \
    --test-conv-indices 8,9 \
    --base-ce "$CE_DIR" \
    --cand-ce "$CE_HN_DIR" \
    --k-values 5,10,20 \
    --max-qa-per-conv 30 2>&1 | tail -25
}

STAGE="${1:-full}"
case "$STAGE" in
  eval-ce)     step_eval_ce ;;
  ft-llm)      step_ft_llm ;;
  ft-emb)      step_ft_emb ;;
  mine-hn)     step_mine_hn ;;
  ft-ce-hn)    step_ft_ce_hn ;;
  eval-ce-hn)  step_eval_ce_hn ;;
  eval-llm)    step_eval_llm ;;
  full)
    # Order: refine CE on hard negatives, then re-eval CE, then LLM FT,
    # then LLM eval (now we know the CE lift), then embedder FT, then
    # final CE eval. Each step is independent enough that a failure
    # doesn't tank the rest.
    step_eval_ce || true
    step_mine_hn || true
    step_ft_ce_hn || true
    step_eval_ce_hn || true
    step_ft_llm || true
    step_eval_llm || true
    step_ft_emb || true
    step_eval_ce || true
    ;;
  hn-only)
    # Just the hard-negative refinement loop (mine → retrain → eval).
    step_mine_hn || exit 1
    step_ft_ce_hn || exit 1
    step_eval_ce_hn || true
    ;;
  llm-only)
    # FT-2 + eval, the two LLM-side steps in sequence.
    step_ft_llm || exit 1
    step_eval_llm || true
    ;;
  *)
    echo "unknown stage: $STAGE"
    echo "stages: eval-ce ft-llm ft-emb mine-hn ft-ce-hn eval-ce-hn"
    echo "        eval-llm full hn-only llm-only"
    exit 1
    ;;
esac
