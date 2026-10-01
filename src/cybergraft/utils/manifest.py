"""Provenance manifest generation for every formal experiment result."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil
import scipy
import yaml

from cybergraft import __version__


def config_hash(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_commit(project_root: Path) -> str:
    """Return the current commit without failing when source is unpacked, not cloned."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_manifest(
    *,
    experiment_id: str,
    config: Dict[str, Any],
    node_count: int,
    edge_count: int,
    project_root: Path,
    mlx_smoke: Dict[str, Any],
    status: str,
    error: Optional[str] = None,
    milestone: str = "M0",
    data_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a serializable manifest for one experiment."""
    manifest: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "milestone": milestone,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(project_root),
        "dataset_versions": {},
        "source_urls": {},
        "query_hashes": [],
        "model_config": config,
        "config_sha256": config_hash(config),
        "random_seed": config["experiment"]["seed"],
        "node_count": node_count,
        "edge_count": edge_count,
        "software_versions": {
            "cybergraft": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
            "psutil": psutil.__version__,
        },
        "mlx_smoke_test": mlx_smoke,
    }
    if data_provenance is not None:
        manifest["dataset_versions"] = data_provenance.get("dataset_versions", {})
        manifest["source_urls"] = data_provenance.get("source_urls", {})
        manifest["query_hashes"] = data_provenance.get("query_hashes", [])
        manifest["data_status"] = data_provenance.get("data_status", {})
        if "license" in data_provenance:
            manifest["license"] = data_provenance["license"]
        if "description" in data_provenance:
            manifest["description"] = data_provenance["description"]
    if error:
        manifest["error"] = error
    return manifest


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Atomically write manifest JSON, avoiding partially-written provenance."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)
