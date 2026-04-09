#!/usr/bin/env python3
"""Shared subprocess runner for consolidated defect_bench pipelines."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    # this file: <repo>/core/runner.py
    return Path(__file__).resolve().parent.parent


def open_dataset_root() -> Path:
    return data_root()


def annotation_source_root() -> Path:
    return repo_root() / "annotation_toolkit" / "src"


def data_root() -> Path:
    return repo_root() / "data_sample"


def vendor_open_dataset_root() -> Path:
    # Backward-compatible alias.
    return data_root()


def run_script(
    script_path: Path,
    extra_args: Iterable[str] | None = None,
    source_root: Path | None = None,
) -> int:
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 2
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(list(extra_args))
    print(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("DEFECT_BENCH_PROJECT_ROOT", str(repo_root()))
    env.setdefault("DEFECT_BENCH_OPEN_DATASET_ROOT", str(source_root or open_dataset_root()))
    proc = subprocess.run(cmd, env=env, cwd=str(script_path.parent))
    return int(proc.returncode)
