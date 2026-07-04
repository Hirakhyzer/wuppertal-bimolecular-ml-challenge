"""Learning-curve entry point with an explicit dataset-mapping guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config, require_confirmed_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare repeated KRR learning-curve experiments.")
    parser.add_argument("--config", default="configs/learning_curve.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    schema = require_confirmed_schema(config)
    raise NotImplementedError(
        "Learning-curve code is scaffolded, but pair-to-XYZ-frame mapping is unresolved. "
        "Review outputs/data_summary.json and configure the real mapping before fitting KRR. Confirmed CSV fields: " + repr(schema)
    )


if __name__ == "__main__":
    main()
