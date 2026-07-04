"""Inspect actual local challenge files without assigning unconfirmed semantics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import inspect_csv, inspect_pair_combinations, validate_raw_inputs, write_json_summary
from src.xyz_parser import atom_count_summary, read_concatenated_xyz


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local bi-molecular challenge dataset.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory holding untracked supplied files.")
    parser.add_argument("--output", default="outputs/data_summary.json", help="Machine-readable JSON summary path.")
    parser.add_argument("--target-column", default=None, help="Confirmed scalar target column; omitted by default.")
    parser.add_argument("--pair-a-column", default=None, help="Confirmed CSV identifier for the A geometry role.")
    parser.add_argument("--pair-b-column", default=None, help="Confirmed CSV identifier for the B geometry role.")
    args = parser.parse_args()

    if (args.pair_a_column is None) != (args.pair_b_column is None):
        parser.error("Provide both --pair-a-column and --pair-b-column, or neither.")

    paths = validate_raw_inputs(args.raw_dir)
    geometries_a = read_concatenated_xyz(paths["Coord_A.xyz"])
    geometries_b = read_concatenated_xyz(paths["Coord_B.xyz"])
    csv_summary = inspect_csv(paths["CouplingEnergies.csv"], target_column=args.target_column)
    summary = {
        "raw_directory": str(Path(args.raw_dir).resolve()),
        "files": {name: {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None} for name, path in paths.items()},
        "coord_a": atom_count_summary(geometries_a),
        "coord_b": atom_count_summary(geometries_b),
        "coupling_csv": csv_summary,
    }
    if paths["Coord_supermol.xyz"].exists():
        summary["coord_supermol"] = atom_count_summary(read_concatenated_xyz(paths["Coord_supermol.xyz"]))
    if args.pair_a_column:
        table = pd.read_csv(paths["CouplingEnergies.csv"])
        summary["pair_combinations"] = inspect_pair_combinations(table, args.pair_a_column, args.pair_b_column)

    write_json_summary(summary, args.output)
    print(f"Wrote inspection summary: {Path(args.output).resolve()}")
    print(f"Coord_A geometries: {summary['coord_a']['geometry_count']}")
    print(f"Coord_B geometries: {summary['coord_b']['geometry_count']}")
    print("CSV columns:", ", ".join(summary["coupling_csv"]["columns"]))
    if args.target_column is None:
        print("No target column was assumed. Confirm it from the summary before running modeling scripts.")


if __name__ == "__main__":
    main()
