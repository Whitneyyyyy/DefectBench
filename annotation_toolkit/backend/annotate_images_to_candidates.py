#!/usr/bin/env python3
"""Compatibility entrypoint for data_sample annotation workflow."""

from runpy import run_module

from defect_bench.annotation_toolkit.src.final_dataset.annotate_images_to_candidates import *  # noqa: F401,F403


if __name__ == "__main__":
    run_module("defect_bench.annotation_toolkit.src.final_dataset.annotate_images_to_candidates", run_name="__main__")

