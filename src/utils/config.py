"""
Configuration loader for experiment YAML files.
"""

import yaml
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "experiment.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load experiment configuration from YAML file.

    Args:
        path: Path to YAML config. Defaults to configs/experiment.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, *keys, default=None) -> Any:
    """Safely get a nested config value.

    Example:
        get_nested(config, 'stage1_recovery', 'timesnet', 'hidden_size')
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
