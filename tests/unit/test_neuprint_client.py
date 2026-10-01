import pytest

from cybergraft.data_ingest.neuprint_client import DataUnavailableError, NeuPrintClient


def test_unavailable_without_token(monkeypatch, tmp_path):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    client = NeuPrintClient(server="https://neuprint.janelia.org", dataset="male-cns:v1.0", token=None)
    assert client.available is False
    assert client.status()["status"] == "unavailable"
    with pytest.raises(DataUnavailableError):
        client.query_neurons({"type": ["GNG232"]})


def test_chunked():
    chunks = list(NeuPrintClient.chunked(list(range(450)), 200))
    assert [len(chunk) for chunk in chunks] == [200, 200, 50]


def test_from_config_reads_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('NEUPRINT_APPLICATION_CREDENTIALS="abc123"\n', encoding="utf-8")
    config = {
        "data": {
            "server": "https://neuprint.janelia.org",
            "dataset": "male-cns:v1.0",
            "timeout_s": 30,
            "max_retries": 3,
        }
    }
    client = NeuPrintClient.from_config(config, project_root=tmp_path)
    assert client.token == "abc123"
    assert client.server == "https://neuprint.janelia.org"
    assert client.dataset == "male-cns:v1.0"
    assert client.timeout_s == 30.0
    assert client.max_retries == 3
