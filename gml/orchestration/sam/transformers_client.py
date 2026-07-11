"""TransformersLLMClient — local HuggingFace-served LLM client.

Same interface as ``HTTPOllamaClient`` / ``LlamaCppClient`` so SAM /
extractor / answer-generator can use a locally-loaded transformers model
(typically a LoRA-fine-tuned Qwen2.5-3B from FT-2) without going through
Ollama.

Why we need this: Ollama can serve GGUF models but our LoRA adapter is
HuggingFace-format. Converting LoRA → merged → GGUF requires
``llama.cpp`` toolchain and adds friction. Loading directly via
transformers is simpler.

Trade-off: slower than llama.cpp/Ollama (no quantization). Reasonable
for evaluating the FT'd model; not for high-throughput production.

Usage:
    from orchestration.sam.transformers_client import TransformersLLMClient
    client = TransformersLLMClient(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path="models/qwen_locomo_ft/adapter",   # LoRA adapter
        device="mps",
    )

Or wire into the factory via GML_LLM_BACKEND=transformers and
GML_TRANSFORMERS_BASE / GML_TRANSFORMERS_ADAPTER env vars.
"""
import asyncio
import os
from typing import Optional

from orchestration.sam._ollama_client import GenerationResult, OllamaClient


# Pre-import bitsandbytes on the main thread when 4-bit mode is requested.
# Why: on Windows, bnb's module-level CUDA init crashes with an access violation
# if first run from a worker thread. ``TransformersLLMClient.generate`` calls
# ``_generate_sync`` via ``asyncio.to_thread``, which runs the import on a
# threadpool worker — too late, segfault. Importing here forces the init to
# happen on the main thread, before any to_thread call can race it.
if os.environ.get("GML_TRANSFORMERS_4BIT", "0") == "1":
    try:
        import bitsandbytes as _bnb  # noqa: F401
    except Exception:
        # Let _ensure_loaded raise the real error if bnb is actually broken.
        pass


# Lazy globals — the model is heavy, so only loaded on first use and
# kept in memory across calls within the same process.
_LOADED: dict[str, object] = {}


def _ensure_loaded(base_model: str, adapter_path: Optional[str], device: str) -> tuple:
    """Load the base model (+ optional LoRA adapter) and tokenizer once.

    Set ``GML_TRANSFORMERS_4BIT=1`` to load the base in nf4 via bitsandbytes —
    required on small GPUs (e.g. the 4 GB RTX 3050) where the fp16 base
    (~6 GB for Qwen2.5-3B) won't fit. CUDA only.
    """
    key = f"{base_model}|{adapter_path}|{device}"
    if key in _LOADED:
        return _LOADED[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_4bit = os.environ.get("GML_TRANSFORMERS_4BIT", "0") == "1" and device == "cuda"
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _LOADED[key] = (tok, model)
    return tok, model


class TransformersLLMClient(OllamaClient):
    """Local transformers-served LLM with the OllamaClient interface."""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "mps",
        max_new_tokens: int = 256,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = device
        self.max_new_tokens = max_new_tokens

    async def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        # JSON mode hint is a soft request — we don't enforce it; the prompt
        # carries the instruction. Many models follow the format anyway.
        return await asyncio.to_thread(
            self._generate_sync, prompt, json_mode, max_tokens, temperature, seed
        )

    def _generate_sync(
        self,
        prompt: str,
        json_mode: bool,
        max_tokens: int | None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        import torch

        tok, model = _ensure_loaded(self.base_model, self.adapter_path, self.device)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: dict = {
            "max_new_tokens": int(max_tokens) if max_tokens is not None else self.max_new_tokens,
            "pad_token_id": tok.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False  # deterministic
        if seed is not None:
            torch.manual_seed(int(seed))

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        # Strip the prompt prefix
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        return GenerationResult(thinking="", answer=text)


def make_transformers_client_from_env() -> Optional[TransformersLLMClient]:
    """Build a TransformersLLMClient from env vars if configured."""
    base = os.environ.get("GML_TRANSFORMERS_BASE")
    if not base:
        return None
    return TransformersLLMClient(
        base_model=base,
        adapter_path=os.environ.get("GML_TRANSFORMERS_ADAPTER") or None,
        device=os.environ.get("GML_TRANSFORMERS_DEVICE", "mps"),
        max_new_tokens=int(os.environ.get("GML_TRANSFORMERS_MAX_TOKENS", "256")),
    )
