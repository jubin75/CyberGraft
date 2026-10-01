from pathlib import Path

from cybergraft.utils.config import load_config


def test_p0_config_inherits_m0_defaults():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "p0_toy.yaml")
    assert config["experiment"]["name"] == "p0-toy-lif"
    assert config["simulation"]["dt_ms"] == 0.5
    assert sum(config["toy_graph"]["layers"]) == 800
