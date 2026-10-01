import numpy as np
import scipy.io

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter


def _write_zebrafish_mat(path, cell_ids, cell_types, connectome):
    n = len(cell_ids)
    cell_id_arr = np.asarray(cell_ids, dtype=np.int32).reshape(n, 1)
    cell_type_obj = np.empty((n, 1), dtype=object)
    for i, label in enumerate(cell_types):
        cell_type_obj[i, 0] = np.array([np.array([label], dtype="<U8")], dtype=object)
    conn = np.asarray(connectome, dtype=np.uint8)
    readme = np.array(["synthetic test"], dtype="<U32")
    dtype = np.dtype([("cellID", "O"), ("cellType", "O"), ("connectome", "O"), ("readme", "O")])
    zc = np.zeros((1, 1), dtype=dtype)
    zc[0, 0]["cellID"] = cell_id_arr
    zc[0, 0]["cellType"] = cell_type_obj
    zc[0, 0]["connectome"] = conn
    zc[0, 0]["readme"] = readme
    scipy.io.savemat(path, {"ZConnectome": zc})


def _small_connectome():
    # 7 cells: 2 _DOs_ (input), 3 _Int_ (lesion), 2 ABD_m (readout)
    cell_ids = [100, 101, 200, 201, 202, 300, 301]
    cell_types = ["_DOs_", "_DOs_", "_Int_", "_Int_", "_Int_", "ABD_m", "ABD_m"]
    conn = np.zeros((7, 7), dtype=np.uint8)
    # _DOs_ -> _Int_
    conn[2, 0] = 1
    conn[3, 0] = 1
    conn[2, 1] = 1
    conn[3, 1] = 1
    # _Int_ -> ABD_m (integrator 2 strong, 3 medium, 4 weak)
    conn[5, 2] = 3
    conn[6, 2] = 2
    conn[5, 3] = 3
    conn[5, 4] = 1
    return cell_ids, cell_types, conn


def test_load_returns_normalized_fields(tmp_path):
    path = tmp_path / "z.mat"
    cell_ids, cell_types, conn = _small_connectome()
    _write_zebrafish_mat(path, cell_ids, cell_types, conn)
    data = ZebrafishAdapter(mat_path=path).load()
    assert data["cell_ids"] == cell_ids
    assert data["cell_types"] == cell_types
    assert data["weights"].shape == (7, 7)
    assert data["weights"].dtype == np.int64
    assert isinstance(data["readme"], str)


def test_load_missing_file_raises(tmp_path):
    adapter = ZebrafishAdapter(mat_path=tmp_path / "missing.mat")
    try:
        adapter.load()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_build_subgraph_groups_and_max_nodes(tmp_path):
    path = tmp_path / "z.mat"
    cell_ids, cell_types, conn = _small_connectome()
    _write_zebrafish_mat(path, cell_ids, cell_types, conn)

    adapter = ZebrafishAdapter(mat_path=path, max_nodes=10)
    graph, provenance = adapter.build_subgraph(
        include_types=["ABD_m", "_Int_", "_DOs_"],
        readout_type="ABD_m",
        input_type="_DOs_",
        lesion_type="_Int_",
    )
    assert graph.node_count == 7
    assert set(graph.groups["input"]) == {0, 1}
    assert set(graph.groups["output"]) == {5, 6}
    assert set(graph.groups["lesion"]) == {2, 3, 4}
    assert provenance["dataset"] == "seung-zebrafish-hindbrain"
    assert provenance["cell_type_counts"]["_DOs_"] == 2
    assert provenance["cell_type_counts"]["_Int_"] == 3


def test_build_subgraph_respects_max_nodes(tmp_path):
    path = tmp_path / "z.mat"
    cell_ids, cell_types, conn = _small_connectome()
    _write_zebrafish_mat(path, cell_ids, cell_types, conn)

    # max_nodes=6 keeps the 4 non-largest cells and only the top 2 _Int_ by weight onto ABD_m.
    adapter = ZebrafishAdapter(mat_path=path, max_nodes=6)
    graph, provenance = adapter.build_subgraph(
        include_types=["ABD_m", "_Int_", "_DOs_"],
        readout_type="ABD_m",
        input_type="_DOs_",
        lesion_type="_Int_",
    )
    assert graph.node_count == 6
    assert len(graph.groups["lesion"]) == 2
    assert provenance["cell_type_counts"]["_Int_"] == 2


def _write_full_zebrafish(tmp_path, full_cell_ids, full_matrix, typed_cell_ids, typed_labels):
    scipy.io.savemat(
        tmp_path / "ConnMatrixPre_cleaned.mat",
        {"ConnMatrixPre_cleaned": np.asarray(full_matrix, dtype=np.uint8)},
    )
    scipy.io.savemat(
        tmp_path / "AllCells.mat",
        {"AllCells": np.asarray(full_cell_ids, dtype=np.int32).reshape(-1, 1)},
    )
    n_typed = len(typed_cell_ids)
    cell_id_arr = np.asarray(typed_cell_ids, dtype=np.int32).reshape(n_typed, 1)
    cell_type_obj = np.empty((n_typed, 1), dtype=object)
    for i, label in enumerate(typed_labels):
        cell_type_obj[i, 0] = np.array([np.array([label], dtype="<U8")], dtype=object)
    readme = np.array(["synthetic full"], dtype="<U32")
    dtype = np.dtype([("cellID", "O"), ("cellType", "O"), ("connectome", "O"), ("readme", "O")])
    zc = np.zeros((1, 1), dtype=dtype)
    zc[0, 0]["cellID"] = cell_id_arr
    zc[0, 0]["cellType"] = cell_type_obj
    zc[0, 0]["connectome"] = np.zeros((n_typed, n_typed), dtype=np.uint8)
    zc[0, 0]["readme"] = readme
    zc_path = tmp_path / "ZConnectome_04292021.mat"
    scipy.io.savemat(zc_path, {"ZConnectome": zc})
    return zc_path


def test_build_full_subgraph(tmp_path):
    # 8 cells: index 0,1 = ABD_m (readout), 2,3 = _Int_ (lesion), 4,5 = _DOs_ (input), 6,7 = untyped.
    full_cell_ids = [100, 101, 200, 201, 300, 301, 400, 401]
    n = len(full_cell_ids)
    W = np.zeros((n, n), dtype=np.uint8)
    W[0, 2] = 1
    W[0, 3] = 1
    W[1, 2] = 1
    W[1, 3] = 1  # _Int_ -> ABD_m
    W[0, 6] = 1
    W[1, 7] = 1  # untyped -> ABD_m
    W[0, 4] = 1
    W[1, 5] = 1  # _DOs_ -> ABD_m
    W[2, 4] = 1
    W[3, 5] = 1  # _DOs_ -> _Int_

    typed_cell_ids = [100, 101, 200, 201, 300, 301]
    typed_labels = ["ABD_m", "ABD_m", "_Int_", "_Int_", "_DOs_", "_DOs_"]

    zc_path = _write_full_zebrafish(tmp_path, full_cell_ids, W, typed_cell_ids, typed_labels)

    adapter = ZebrafishAdapter(mat_path=zc_path, max_nodes=700)
    graph, provenance = adapter.build_full_subgraph(
        readout_type="ABD_m", lesion_type="_Int_", input_type="_DOs_", max_nodes=700
    )

    assert graph.node_count == 8
    assert set(graph.groups["output"]) == {0, 1}
    assert set(graph.groups["lesion"]) == {2, 3}
    assert set(graph.groups["input"]) == {4, 5}

    untyped = [node for node in graph.nodes if node.cell_type == "untyped"]
    assert len(untyped) == 2
    assert set(graph.groups["lesion"]) & {node.id for node in untyped} == set()

    assert provenance["dataset"] == "seung-zebrafish-hindbrain-full"
    assert provenance["cell_type_counts"]["untyped"] == 2
    assert provenance["cell_type_counts"]["ABD_m"] == 2
