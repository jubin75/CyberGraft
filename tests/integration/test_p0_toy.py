from pathlib import Path

from cybergraft.sim.runner import run_experiment
from cybergraft.utils.config import load_config


def test_p0_toy_creates_metrics_and_manifest(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "p0_toy.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "results")
    result_dir, metrics = run_experiment(config, project_root=root)
    assert metrics["passed"] is True
    assert metrics["output_spikes"] >= 1
    assert (result_dir / "metrics.json").is_file()
    assert (result_dir / "manifest.json").is_file()
    assert (result_dir / "activity.npz").is_file()
