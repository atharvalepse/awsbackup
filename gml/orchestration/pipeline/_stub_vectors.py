"""Shared deterministic-hash-to-vector helper used by stub embedder and
stub retriever so the same text produces the same vector in both places.

Pure utility — not part of any pipeline stage. Real embedders and real
vector stores will produce vectors via their own APIs and won't touch this.
"""
import hashlib
import math
import struct


def hash_to_unit_vector(text: str, dim: int) -> list[float]:
    """Deterministic L2-normalized vector derived from SHA-256 of ``text``.

    Same text always yields the same vector. Identical to the algorithm used
    by both the stub embedder and the stub retriever's record-side embedding,
    so cosine similarity between a stub-embedded query and a stub-retriever
    record is mathematically meaningful.
    """
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 4:
        h = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
        raw.extend(h)
        counter += 1
    ints = struct.unpack(f"<{dim}i", bytes(raw[: dim * 4]))
    floats = [i / 2_147_483_647.0 for i in ints]
    norm = math.sqrt(sum(x * x for x in floats))
    if norm == 0.0:
        return floats
    return [x / norm for x in floats]
