#!/usr/bin/env python3
"""Resolve current commits for ``[tool.uv.sources]`` git deps.

Install is still ``uv pip install .`` from pyproject.docker. Docker caches that
RUN even when a branch pin (``rev = "fix-git-pull"``) moves, so builds pass the
resolved SHAs as ``--build-arg UV_GIT_SHAS=...`` to invalidate the layer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path


def _ls_remote(url: str, rev: str) -> str:
    refs = [rev]
    if rev != "HEAD" and not rev.startswith("refs/"):
        refs.extend([f"refs/heads/{rev}", f"refs/tags/{rev}"])
    for ref in refs:
        result = subprocess.run(
            ["git", "ls-remote", url, ref],
            check=False,
            capture_output=True,
            text=True,
        )
        lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
        if lines:
            return lines[0].split()[0]
    raise SystemExit(f"Could not resolve {url}@{rev}")


def resolve(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    parts: list[str] = []
    for name, spec in sources.items():
        if not isinstance(spec, dict) or "git" not in spec:
            continue
        url = str(spec["git"])
        rev = str(spec.get("rev") or spec.get("branch") or spec.get("tag") or "HEAD")
        parts.append(f"{name}={_ls_remote(url, rev)}")
    if not parts:
        raise SystemExit(f"No git sources in {pyproject}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", type=Path)
    args = parser.parse_args()
    sys.stdout.write(resolve(args.pyproject))


if __name__ == "__main__":
    main()
