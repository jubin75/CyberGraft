import json
from pathlib import Path

import numpy as np
import scipy.io

from cybergraft.experiment.bypass_experiment import run_synapse_weighted_rstdp
from cybergraft.utils.config import load_config


def _write_full_mat(tmp_path, cell_ids, full_matrix, typed_cell_ids, typed_labels):
    scipy.io.savemat(
        tmp_path / "ConnMatrixPre_cleaned.mat",
        {"ConnMatrixPre_cleaned": np.asarray(full_matrix, dtype=np.uint8)},
    )
    scipy.io.savemat(
        tmp_path / "AllCells.mat",
        {"AllCells": np.asarray(cell_ids, dtype=np.int32).reshape(-1, 1)},
    )
    n_typed = len(typed_cell_ids)
    cell_id_arr = np.asarray(typed_cell_ids, dtype=np.int32).reshape(n_typed, 1)
    cell_type_obj = np.empty((n_typed, 1), dtype=object)
    for i, label in enumerate(typed_labels):
        cell_type_obj[i, 0] = np.array([np.array([label], dtype="<U8")], dtype=object)
    readme = np.array(["synthetic sw"], dtype="<U32")
    dtype = np.dtype([("cellID", "O"), ("cellType", "O"), ("connectome", "O"), ("readme", "O")])
    zc = np.zeros((1, 1), dtype=dtype)
    zc[0, 0]["cellID"] = cell_id_arr
    zc[0, 0]["cellType"] = cell_type_obj
    zc[0, 0]["connectome"] = np.zeros((n_typed, n_typed), dtype=np.uint8)
    zc[0, 0]["readme"] = readme
    zc_path = tmp_path / "ZConnectome_04292021.mat"
    scipy.io.savemat(zc_path, {"ZConnectome": zc})
    return zc_path


def test_p8_sw_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "p8_sw.yaml")

    # 9 cells: index 0,1=_DOs_, 2,3,4=_Int_, 5,6=ABD_m, 7,8=ABD_i.
    cell_ids = [100, 101, 200, 201, 202, 300, 301, 400, 401]
    n = len(cell_ids)
    W = np.zeros((n, n), dtype=np.uint8)
    W[2, 0] = 3
    W[3, 0] = 3
    W[4, 1] = 3
    W[2, 1] = 3  # _DOs_ -> _Int_
    W[5, 2] = 4
    W[6, 2] = 4
    W[5, 3] = 4
    W[6, 4] = 4  # _Int_ -> ABD_m
    W[5, 0] = 2
    W[6, 1] = 2  # _DOs_ -> ABD_m direct
    W[5, 7] = 5
    W[6, 7] = 5
    W[5, 8] = 5
    W[6, 8] = 5  # ABD_i -> ABD_m (bypass)
    W[7, 2] = 4
    W[8, 3] = 4  # _Int_ -> ABD_i

    typed_labels = ["_DOs_", "_DOs_", "_Int_", "_Int_", "_Int_", "ABD_m", "ABD_m", "ABD_i", "ABD_i"]
    zc_path = _write_full_mat(tmp_path, cell_ids, W, cell_ids, typed_labels)

    config["experiment"]["output_dir"] = str(tmp_path / "results")
    config["data"]["mat_file"] = str(zc_path)
    config["mapping"]["n_episodes"] = 5
    config["stimulus"] = {
        "type": "rhythmic",
        "target_group": "input",
        "amplitude": 300.0,
        "frequency_hz": 5.0,
        "start_ms": 20.0,
        "duration_ms": 360.0,
    }

    result_dir, metrics = run_synapse_weighted_rstdp(config, project_root=root, mat_path=zc_path)

    assert metrics["milestone"] == "M8"
    for key in ("plain_rstdp", "sw_rstdp", "supervised"):
        assert "recovery" in metrics[key]
        assert "gain" in metrics[key]
    assert (result_dir / "metrics.json").is_file()
    assert (result_dir / "manifest.json").is_file()
    assert (result_dir / "synapse_weighted.png").is_file()

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["milestone"] == "M8"
