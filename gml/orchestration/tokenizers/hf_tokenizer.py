"""HuggingFace `tokenizers`-backed tokenizer for any model with a
public ``tokenizer.json`` on the Hub.

This is the real-tokenizer path for Claude / Llama / DeepSeek — accurate
token counts via the same tokenizer the target model uses internally.

Tokenizer files are downloaded once via ``hf_hub_download`` and cached
locally under ``~/.cache/huggingface/`` (HF default) on first use; later
runs are local-only.

Falls back gracefully: if the download fails (no network, gated repo) the
constructor raises so callers can use a tiktoken approximation.
"""
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

from orchestration.tokenizers.base import Tokenizer


# ---------------------------------------------------------------------------
# Known good public tokenizer-only repos. These ship just tokenizer.json,
# tokenizer_config.json, etc. — no model weights — so the download is small
# (single-digit MB).
# ---------------------------------------------------------------------------

KNOWN_REPOS: dict[str, str] = {
    # Llama 3 tokenizer (BPE). Defaults are open mirrors so no HF login is
    # required; ``meta-llama/*`` repos are gated and need
    # ``huggingface-cli login`` first. Override the constructor's ``repo_id``
    # to use the exact Meta repo for byte-perfect tokenization.
    "llama-3":   "NousResearch/Meta-Llama-3-8B",
    "llama-3.1": "NousResearch/Meta-Llama-3.1-8B",
    "llama-3.2": "NousResearch/Meta-Llama-3-8B",  # same tokenizer family
    # DeepSeek R1 family (Qwen-derived BPE) — public, no auth
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    # Claude has no public tokenizer; ClaudeAPITokenizer handles that.
}


def _download_tokenizer(repo_id: str) -> Path:
    """Pull ``tokenizer.json`` from a HF repo into the local cache and
    return the on-disk path.

    Uses ``huggingface_hub`` which ships transitively with ``fastembed``.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    return Path(path)


class HFTokenizerWrapper(Tokenizer):
    """Real tokenizer for any model identified by its HF repo id.

    Args:
        repo_id: HuggingFace repo (e.g. ``"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"``).
        version_tag: short label for ``Tokenizer.version`` (e.g. ``"deepseek-r1"``).
    """

    def __init__(self, repo_id: str, version_tag: str | None = None) -> None:
        self.repo_id = repo_id
        self._version_tag = version_tag or repo_id.split("/")[-1].lower()
        path = _download_tokenizer(repo_id)
        self._tokenizer = HFTokenizer.from_file(str(path))

    @property
    def version(self) -> str:
        return f"hf:{self._version_tag}"

    def count(self, text: str) -> int:
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return len(enc.ids)


class RealLlamaTokenizer(HFTokenizerWrapper):
    """Llama 3.x tokenizer (BPE). Uses an open mirror by default so no HF
    login is required; pass an explicit ``repo_id`` to use ``meta-llama/*``
    after ``huggingface-cli login`` for byte-perfect Meta tokenization."""

    def __init__(self, repo_id: str = KNOWN_REPOS["llama-3"]) -> None:
        super().__init__(repo_id=repo_id, version_tag="llama-3")


class RealDeepSeekTokenizer(HFTokenizerWrapper):
    """DeepSeek R1 tokenizer. Default repo: DeepSeek-R1-Distill-Llama-8B."""

    def __init__(self, repo_id: str = KNOWN_REPOS["deepseek-r1"]) -> None:
        super().__init__(repo_id=repo_id, version_tag="deepseek-r1")
