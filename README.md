# Wuppertal Bi-Molecular ML Challenge

A rigorous, reproducible Python research repository for predicting a scalar excitonic-coupling target from **pairs of molecular geometries**.

> **Data boundary:** raw challenge files are expected locally in `data/raw/` and are intentionally excluded from version control. This repository contains no raw dataset, no precomputed results, and no fabricated figures or metrics.

## Objective

Build and benchmark a bi-molecular regression workflow using Coulomb-matrix molecular representations and Kernel Ridge Regression (KRR). The workflow is designed to inspect the actual supplied files before making dataset-specific choices.

## Scientific safeguards

- Dataset schema, atom counts, units, target column, and pair identifiers are never assumed in code.
- All random operations use configurable fixed seeds.
- Hyperparameter selection is restricted to training/validation data.
- A final test set remains untouched until final evaluation.
- Two split strategies are supported and documented: random pair split and geometry-aware split.
- Learning-curve figures and report inputs are generated only after local experiments run.
- Kernel Ridge Regression is protected by configurable training-size limits because exact kernel methods can become computationally expensive.

## Repository structure

```text
README.md
requirements.txt
.gitignore
LICENSE
data/                  Dataset placement and data dictionary guidance
docs/                  Methodology, prompt history, and LaTeX report template
configs/               Dataset-agnostic experiment settings
src/                   Reusable research code
scripts/               Runnable inspection and experiment entry points
notebooks/             Guided analysis notebooks
tests/                 Unit tests using only synthetic toy geometries
outputs/               Local generated summaries, results, and figures
```

## Expected local dataset placement

Place the supplied files **locally** here:

```text
data/raw/
  Coord_A.xyz
  Coord_B.xyz
  CouplingEnergies.csv
  Coord_supermol.xyz   # optional; not required by the initial workflow
```

Read [`data/README.md`](data/README.md) before copying files. Do not commit `data/raw/`.

## Installation

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## First command: inspect the actual data

```bash
python scripts/inspect_data.py
```

This command parses both concatenated XYZ files, reads the CSV header and types, reports file and geometry dimensions, checks missing and duplicate rows, summarizes numeric columns without guessing which column is the target, and writes:

```text
outputs/data_summary.json
```

For pair-combination and target-specific diagnostics, rerun after inspection with the confirmed column names:

```bash
python scripts/inspect_data.py \
  --target-column <confirmed_target_column> \
  --pair-a-column <confirmed_A_identifier_column> \
  --pair-b-column <confirmed_B_identifier_column>
```

## Planned experiment commands

These scripts deliberately stop with a clear schema-confirmation message until `configs/baseline_krr.yaml` has the real CSV column names and the inspection summary has been reviewed.

```bash
python scripts/run_baseline.py --config configs/baseline_krr.yaml
python scripts/run_learning_curve.py --config configs/learning_curve.yaml
python scripts/run_final_experiment.py --config configs/baseline_krr.yaml
```

## Modeling approach

1. Parse individual structures from concatenated XYZ files.
2. Build Coulomb matrices using
   \(C_{ii}=0.5 Z_i^{2.4}\) and \(C_{ij}=Z_i Z_j / ||R_i-R_j||\).
3. Apply a documented canonical row/column ordering and consistent padding.
4. Build pair features using either:
   - `directed_concatenation = [A, B]`
   - `interaction_features = [A, B, |A-B|, A*B]`
5. Train `Pipeline(StandardScaler(), KernelRidge(kernel="rbf"))`.
6. Select `alpha` and `gamma` only with training/validation data.
7. Report MAE as the primary metric.
8. Generate a repeated, double-logarithmic validation learning curve.

The target is **not assumed to be symmetric** under swapping molecular roles. A reverse-pair symmetry diagnostic is included but will run only after the actual CSV pair columns are confirmed.

## Split strategies

| Split | Meaning | Main limitation |
| --- | --- | --- |
| Random pair split | Randomly splits pair rows into train, validation, and test sets. | The same molecular geometry may occur in both train and test pairs, so it can be optimistic. |
| Geometry-aware split | Partitions unique geometry identifiers and retains only within-partition pairs. | Cross-partition pairs are dropped; fewer samples may remain, but held-out geometries cannot appear in training. |

Details are in [`docs/methodology.md`](docs/methodology.md).

## Outputs created only after local execution

```text
outputs/data_summary.json
outputs/results/*.json
outputs/results/learning_curve_results.csv
outputs/figures/learning_curve_loglog.png
outputs/figures/learning_curve_loglog.pdf
```

No numerical outcomes are included in this repository until generated from your local dataset.

## Reproducibility

- Fixed seeds are declared in YAML configuration.
- Configurations are copied into structured run metadata by experiment code.
- Unit tests validate XYZ parsing, Coulomb matrices, pair representations, and leakage-resistant split logic.
- `docs/LLM_PROMPT_HISTORY.md` preserves the initial build prompt and is intended for manually added future interactions.

## Final report

`docs/report_template.tex` is a one-page placeholder-only LaTeX template. Populate it only with figures, settings, and metrics generated by executed experiments.

## Limitations

This initial scaffold does not infer the CSV target or pair-ID columns. Those require inspection of the supplied challenge files. It also does not blindly fit KRR to a potentially large full-pair dataset; kernel-method sample limits must be chosen after assessing available memory and runtime.
