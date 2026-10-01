import json
from pathlib import Path

import numpy as np
import scipy.io

from cybergraft.experiment.lesion_experiment import run_lesion_experiment
from cybergraft.utils.config import load_config


def _write_zebrafish_mat(path):
    cell_ids = [10, 11, 20, 21, 22, 30, 31, 40, 41]
    cell_types = ["_DOs_", "_DOs_", "_Int_", "_Int_", "_Int_", "ABD_m", "ABD_m", "ABD_i", "ABD_i"]
    n = len(cell_ids)
    conn = np.zeros((n, n), dtype=np.uint8)
    # _DOs_ -> _Int_
    conn[2, 0] = 3
    conn[3, 0] = 3
    conn[2, 1] = 3
    conn[4, 1] = 3
    # _Int_ -> ABD_m (readout)
    conn[5, 2] = 4
    conn[6, 2] = 4
    conn[5, 3] = 4
    conn[6, 4] = 4
    # _Int_ -> ABD_i (dead-end, not readout)
    conn[7, 2] = 4
    conn[8, 3] = 4

    cell_id_arr = np.asarray(cell_ids, dtype=np.int32).reshape(n, 1)
    cell_type_obj = np.empty((n, 1), dtype=object)
    for i, label in enumerate(cell_types):
        cell_type_obj[i, 0] = np.array([np.array([label], dtype="<U8")], dtype=object)
    readme = np.array(["synthetic integration"], dtype="<U32")
    dtype = np.dtype([("cellID", "O"), ("cellType", "O"), ("connectome", "O"), ("readme", "O")])
    zc = np.zeros((1, 1), dtype=dtype)
    zc[0, 0]["cellID"] = cell_id_arr
    zc[0, 0]["cellType"] = cell_type_obj
    zc[0, 0]["connectome"] = conn
    zc[0, 0]["readme"] = readme
    scipy.io.savemat(path, {"ZConnectome": zc})


def test_p2_lesion_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "p2_lesion.yaml")
    mat_path = tmp_path / "z.mat"
    _write_zebrafish_mat(mat_path)
    config["experiment"]["output_dir"] = str(tmp_path / "results")
    config["data"]["mat_file"] = str(mat_path)
    # The spec's Poisson amplitude (25) is below the reference-LIF firing
    # threshold for intermittent input; use a strong amplitude so the synthetic
    # chain actually propagates and the lesion comparison is meaningful.
    config["stimulus"]["amplitude"] = 300.0

    result_dir, metrics = run_lesion_experiment(config, project_root=root, mat_path=mat_path)

    assert metrics["passed"] is True
    assert metrics["milestone"] == "M2"
    assert metrics["lesion_severity_mean"] > 0
    assert metrics["intact_output_spikes"] > metrics["lesioned_output_spikes"]

    assert (result_dir / "metrics.json").is_file()
    assert (result_dir / "manifest.json").is_file()

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["milestone"] == "M2"
