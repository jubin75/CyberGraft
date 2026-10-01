import json
from pathlib import Path

from cybergraft.data_ingest.feeding_experiment import run_feeding_experiment
from cybergraft.utils.config import load_config

from fakes import feeding_backend


def test_p1_feeding_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "p1_feeding.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "results")
    config["data"]["cache_dir"] = str(tmp_path / "cache")

    result_dir, metrics = run_feeding_experiment(config, project_root=root, client=feeding_backend())

    assert metrics["passed"] is True
    assert metrics["output_spikes"] >= 1
    assert metrics["milestone"] == "M1"

    assert (result_dir / "metrics.json").is_file()
    assert (result_dir / "manifest.json").is_file()
    assert (result_dir / "activity.npz").is_file()

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["milestone"] == "M1"
    assert manifest["data_status"]["available"] is True
