"""Run one pytest node under a hard cap and emit a machine-readable status."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time

CORE_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pytest",
    "coverage",
    "ruff",
    "mypy",
    "mkdocs",
    "setuptools",
    "setuptools-scm",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in CORE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_bounded_test(node_id: str, timeout_seconds: float) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", node_id, "--no-cov"]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        status = "unresolved-timeout"
        return_code = None
        stdout = _decoded(exc.stdout)
        stderr = _decoded(exc.stderr)
    else:
        status = "passed" if completed.returncode == 0 else "failed"
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    return {
        "node_id": node_id,
        "status": status,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": package_versions(),
        },
    }


def _decoded(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = run_bounded_test(args.node_id, args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
