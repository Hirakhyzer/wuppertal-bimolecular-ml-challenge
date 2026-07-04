"""Fixed-test final-evaluation entry point with an explicit schema guard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config, require_confirmed_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final held-out KRR evaluation.")
    parser.add_argument("--config", default="configs/baseline_krr.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    schema = require_confirmed_schema(config)
    raise NotImplementedError(
        "Final evaluation is intentionally unavailable before inspecting the actual dataset and implementing its confirmed "
        "CSV-to-XYZ-frame mapping. This preserves the fixed held-out test set from accidental misuse. Confirmed fields: " + repr(schema)
    )


if __name__ == "__main__":
    main()
