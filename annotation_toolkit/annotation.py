#!/usr/bin/env python3
"""Entry point for the data_sample annotation workflow."""

from __future__ import annotations

import argparse

from defect_bench.core.runner import annotation_source_root, data_root, run_script


def main() -> int:
    parser = argparse.ArgumentParser(description="Run data_sample annotation workflow.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to canonical script")
    args = parser.parse_args()
    script = annotation_source_root() / "final_dataset" / "annotate_images_to_candidates.py"
    return run_script(script, args.extra, source_root=data_root())


if __name__ == "__main__":
    raise SystemExit(main())
