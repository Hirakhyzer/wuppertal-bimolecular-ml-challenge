# Local dataset placement

This repository does **not** contain the challenge dataset. Keep all supplied raw files local and untracked.

Create this directory on your machine after cloning:

```text
data/raw/
```

Place the provided files there using these exact names:

```text
data/raw/Coord_A.xyz
data/raw/Coord_B.xyz
data/raw/CouplingEnergies.csv
data/raw/Coord_supermol.xyz   # optional
```

## What the first inspection does

Run:

```bash
python scripts/inspect_data.py
```

The inspection script reads the actual files and writes `outputs/data_summary.json`. It will determine, rather than assume:

- the CSV header, data types, dimensions, missing values, duplicate rows, and numeric-column summaries;
- the number of concatenated XYZ geometries in each coordinate file;
- atom-count distributions for both coordinate files;
- whether the optional supermolecule file is present;
- pair-combination and target-distribution checks once confirmed CSV column names are supplied.

## Data handling rules

- Do not commit `data/raw/` unless you have explicit permission to do so.
- Do not rename files without updating the inspection command or configuration.
- Do not infer target units, target-column meaning, or pair-index mapping from file names alone.
- Keep any generated local CSV or cache files out of version control unless they are explicitly approved non-sensitive derivatives.
