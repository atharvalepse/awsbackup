import tomllib
from pathlib import Path

from orchestration.pipeline.contracts import OrchestrationConfig


def load_config(path: str | Path) -> OrchestrationConfig:
    """Load an OrchestrationConfig from a TOML file at ``path``."""
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)
    return OrchestrationConfig(**data)
