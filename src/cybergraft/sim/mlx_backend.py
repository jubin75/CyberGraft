"""Optional MLX smoke test.

M0 keeps this separate from the correctness backend: lack of MLX, a missing
Metal device, or an MLX runtime crash must never prevent the reference backend
or its tests from running.
"""

import importlib.util
import subprocess
import sys
from typing import Any, Dict


def run_mlx_smoke_test() -> Dict[str, Any]:
    """Run a tiny MLX operation in a child process.

    Some headless or virtualized macOS sessions abort the interpreter while
    loading Metal, rather than raising a Python exception.  A subprocess keeps
    that platform failure out of the experiment process and makes the result
    explicit in the manifest.
    """
    if importlib.util.find_spec("mlx") is None:
        return {
            "installed": False,
            "available": False,
            "status": "skipped",
            "reason": "mlx is not installed",
        }
    code = "import mlx.core as mx; print(float(mx.sum(mx.array([1., 2., 3.])).item()))"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "installed": True,
            "available": False,
            "status": "unavailable",
            "reason": "MLX smoke test timed out while initializing a device",
        }
    if completed.returncode == 0:
        return {
            "installed": True,
            "available": True,
            "status": "passed",
            "sum": float(completed.stdout.strip()),
        }
    detail = (completed.stderr or completed.stdout or "MLX process exited without detail").strip()
    return {
        "installed": True,
        "available": False,
        "status": "unavailable",
        "return_code": completed.returncode,
        "reason": detail[-500:],
    }
