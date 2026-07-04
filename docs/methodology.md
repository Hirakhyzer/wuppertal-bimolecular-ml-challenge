# Methodology

## Scope and data boundary

This repository is intentionally dataset-agnostic until `data/raw/Coord_A.xyz`, `data/raw/Coord_B.xyz`, and `data/raw/CouplingEnergies.csv` are inspected locally. No source file names are treated as evidence of CSV column semantics, molecular counts, units, or target symmetry.

## 1. Concatenated XYZ parsing

A concatenated XYZ file is treated as a sequence of frames. Each frame begins with an integer atom count, followed by one comment line and exactly that many atom records. Every atom record must contain a chemical symbol and three Cartesian coordinates. The parser preserves frame order and records source comments for traceability.

The inspection command reports the number of parsed geometries and atom-count distributions. It fails loudly on malformed frame boundaries instead of silently dropping records.

## 2. Coulomb-matrix molecular representation

For a molecule with atomic numbers \(Z_i\) and Cartesian coordinates \(R_i\), the Coulomb matrix is

\[
C_{ii}=0.5Z_i^{2.4},\qquad
C_{ij}=\frac{Z_iZ_j}{\lVert R_i-R_j\rVert}\quad(i\ne j).
\]

The implementation rejects coincident distinct atoms because the off-diagonal denominator would be zero.

### Canonical ordering and padding

Rows and columns are reordered together using descending row norms, with a permutation-invariant row-signature tie-breaker. This provides a practical ordering strategy for fixed-length learning features. Matrices are padded to the maximum atom count **in the training representation fit**. The upper-triangular entries, including the diagonal, are flattened because the matrix is symmetric.

Row-norm ordering is a representation choice rather than a proof of complete molecular uniqueness. Enantiomers and other physically distinct situations may still share a descriptor; this limitation must be stated in the final report.

## 3. Pair representation

Let \(a\) and \(b\) be fixed-length molecular feature vectors for roles A and B.

- `directed_concatenation`: \([a,b]\)
- `interaction_features`: \([a,b,|a-b|,a\odot b]\)

Both retain role direction. The code does not symmetrize pairs. Only after inspecting reverse-pair rows and target consistency may a separate symmetrized experimental condition be justified.

## 4. Kernel Ridge Regression

The primary estimator is:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("krr", KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)),
])
```

`alpha` controls regularization and `gamma` controls RBF width. A configurable grid search uses only training and validation partitions. The held-out test partition is not used for hyperparameter selection.

Exact RBF Kernel Ridge Regression needs a dense kernel matrix and can have memory/time costs that grow approximately quadratically with the number of training pairs. The scripts enforce a configurable maximum KRR training size and record the chosen size. Gaussian Process Regression is reserved as an optional future comparison because it has related scaling constraints.

## 5. Split strategies and leakage control

### Random pair baseline

Pair rows are randomly partitioned into training, validation, and test sets. This is useful as a baseline but can place a molecular geometry in both training and test pairs. Its test MAE may therefore be optimistic for generalization to unseen geometries.

### Geometry-aware split

Unique geometry identifiers are split into train, validation, and test identifier sets. A pair is retained only when both of its geometries lie in the same partition; cross-partition pairs are explicitly dropped. This provides a stricter evaluation because test geometries cannot occur in training pairs. The cost is reduced sample count, especially when the observed pair table is dense across geometry combinations.

The inspection stage must confirm how CSV rows reference frames in `Coord_A.xyz` and `Coord_B.xyz` before a geometry-aware split is run.

## 6. Evaluation

The primary metric is mean absolute error:

\[
\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|.
\]

The final experiment uses the single fixed test set created from the configured seed and split method. Validation MAE selects hyperparameters; test MAE is reported only after selection is fixed.

## 7. Learning curve

Learning curves use logarithmically spaced feasible training sizes, capped by available training pairs and the configured KRR limit. Each size is repeated over multiple fixed seeds where feasible. The output records mean validation MAE and standard deviation, and visualizes the relationship on double-logarithmic axes:

- x-axis: number of training bi-molecular samples/pairs;
- y-axis: validation MAE.

The final figure is written only after code executes locally:

```text
outputs/figures/learning_curve_loglog.png
outputs/figures/learning_curve_loglog.pdf
```

## 8. Limitations requiring transparent reporting

- Dataset units and target semantics are unknown until the CSV is inspected.
- Pair direction and any A/B exchange symmetry must be empirically assessed.
- Coulomb matrices are not guaranteed to be uniquely identifying descriptors.
- Random pair splits may overestimate transfer performance.
- Geometry-aware splits may drop substantial numbers of cross-partition pairs.
- Exact KRR can become computationally impractical at large pair counts.
- No result, figure, or report table should be presented until generated by an executed experiment.
