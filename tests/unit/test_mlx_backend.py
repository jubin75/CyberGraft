from cybergraft.sim.mlx_backend import run_mlx_smoke_test


def test_mlx_smoke_test_is_nonfatal_when_unavailable():
    result = run_mlx_smoke_test()
    assert result["status"] in {"passed", "skipped", "unavailable"}
    assert "available" in result
