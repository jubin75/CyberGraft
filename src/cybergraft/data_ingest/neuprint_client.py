"""Lazy, resilient neuPrint client wrapper for MaleCNS data access."""

import importlib.util
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


class DataUnavailableError(RuntimeError):
    """Raised when a neuPrint query cannot run because the client is unavailable."""


def _normalize(value: Any) -> Any:
    """Recursively convert numpy scalars and arrays to plain Python values."""
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_env_token(env_path: Path) -> Optional[str]:
    """Parse a ``NEUPRINT_APPLICATION_CREDENTIALS=<value>`` line from ``.env``."""
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("NEUPRINT_APPLICATION_CREDENTIALS"):
            continue
        _, separator, value = stripped.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        return value
    return None


class NeuPrintClient:
    MAX_BATCH = 200

    def __init__(
        self,
        *,
        server: str,
        dataset: str,
        token: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.server = server
        self.dataset = dataset
        self.token = token or None
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = None
        self._reason = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if importlib.util.find_spec("neuprint") is None:
            self._reason = "neuprint-python not installed"
            return False
        if not self.token:
            self._reason = "no neuPrint token (set NEUPRINT_APPLICATION_CREDENTIALS)"
            return False
        try:
            from neuprint import Client

            client = Client(self.server, self.dataset, token=self.token)
            self._mount_timeout(client)
            self._client = client
            return True
        except Exception as exc:
            self._reason = f"neuPrint client unavailable: {type(exc).__name__}"
            return False

    @property
    def available(self) -> bool:
        return self._ensure_client()

    def status(self) -> dict:
        installed = importlib.util.find_spec("neuprint") is not None
        available = self.available
        return {
            "installed": installed,
            "available": available,
            "status": "passed" if available else "unavailable",
            "reason": self._reason,
            "server": self.server,
            "dataset": self.dataset,
        }

    def _mount_timeout(self, client) -> None:
        """Mount (connect, read) timeouts on the client session without raising."""
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            class _TimeoutAdapter(HTTPAdapter):
                def __init__(self, timeout, *args, **kwargs):
                    self._timeout = timeout
                    super().__init__(*args, **kwargs)

                def send(self, request, **kwargs):
                    if kwargs.get("timeout") is None:
                        kwargs["timeout"] = getattr(self, "_timeout", (60.0, 60.0))
                    return super().send(request, **kwargs)

            timeout = (self.timeout_s, self.timeout_s)
            adapter = _TimeoutAdapter(
                timeout,
                max_retries=Retry(
                    total=self.max_retries,
                    backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                ),
            )
            client.session.mount("https://", adapter)
            client.session.mount("http://", adapter)
        except Exception:
            pass

    def query_neurons(self, criteria: dict) -> list[dict]:
        if not self.available:
            raise DataUnavailableError(self._reason)
        from neuprint import fetch_neurons, NeuronCriteria as NC

        result = fetch_neurons(NC(**criteria), omit_rois=True, client=self._client)
        if isinstance(result, tuple):
            result = result[0]
        return _normalize(result.to_dict("records"))

    def query_connections(
        self, sources: Optional[list[int]], targets: Optional[list[int]], min_weight: int = 1
    ) -> list[dict]:
        if not self.available:
            raise DataUnavailableError(self._reason)
        if sources == [] or targets == []:
            return []
        from neuprint import fetch_adjacencies

        _, roi_conn_df = fetch_adjacencies(
            sources=sources,
            targets=targets,
            omit_rois=True,
            min_total_weight=min_weight,
            client=self._client,
        )
        records = _normalize(roi_conn_df.to_dict("records"))
        return [
            {
                "bodyId_pre": int(record["bodyId_pre"]),
                "bodyId_post": int(record["bodyId_post"]),
                "weight": int(record["weight"]),
            }
            for record in records
        ]

    @staticmethod
    def from_config(config: dict, project_root: Path) -> "NeuPrintClient":
        data = config["data"]
        token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        if not token:
            token = _read_env_token(Path(project_root) / ".env")
        return NeuPrintClient(
            server=data["server"],
            dataset=data["dataset"],
            token=token,
            timeout_s=float(data.get("timeout_s", 60.0)),
            max_retries=int(data.get("max_retries", 2)),
        )

    @staticmethod
    def chunked(seq, n: int = MAX_BATCH) -> Iterator[list]:
        for index in range(0, len(seq), n):
            yield seq[index:index + n]
