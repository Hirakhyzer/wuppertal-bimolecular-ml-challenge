"""Baseline KRR entry point.

This scaffold intentionally refuses to guess how CSV rows reference XYZ frames.
After the inspection summary is reviewed, implement the confirmed mapping in the
configuration and dataset-assembly layer before enabling this experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config, require_confirmed_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the baseline KRR experiment.")
    parser.add_argument("--config", default="configs/baseline_krr.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    schema = require_confirmed_schema(config)
    raise NotImplementedError(
        "Schema is confirmed as " + repr(schema) + ". Next, map those CSV identifiers to parsed XYZ frame indices "
        "using the actual inspection result. This deliberate stop prevents invented geometry-index assumptions."
    )


if __name__ == "__main__":
    main()
