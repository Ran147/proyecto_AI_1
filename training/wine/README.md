# Wine quality (`wine`)

## Problem

- Task: multiclass classification (`low` = score 3-5, `medium` = 6,
  `high` = 7-9)
- What it answers: the expected sensory quality of a wine from its
  physicochemical properties.
- Voice commands (Spanish): "JarvisTEC, calidad del vino", "JarvisTEC,
  clasificar vino", "JarvisTEC, qué tan bueno es este vino"

## Dataset

- Source: Kaggle
  [rajyellow46/wine-quality](https://www.kaggle.com/datasets/rajyellow46/wine-quality),
  a version with missing values of the *Wine Quality* dataset by Cortez et al.
  (2009), UCI Machine Learning Repository.
- License: the original UCI dataset is published under CC BY 4.0; check the
  license shown on the Kaggle page.
- Expected file: `data/wine_quality/winequalityN.csv`
- 6497 rows x 13 columns (1599 red, 4898 white); 38 missing values in 34 rows;
  1168 duplicated rows (removed: 5329 remain).
- Target: `quality_class`, derived from `quality`. `quality` is not an input.
- Columns are renamed to lowercase snake_case (`pH` becomes `ph`); they are
  the API input fields.

## Configuration

- `SEED = 42`, stratified 80/20 split (4263 train / 1066 test), stratified
  5-fold cross-validation.
- Preprocessing inside the `Pipeline`: median + `StandardScaler` (numeric),
  mode + `OneHotEncoder` (`type`).
- Model: `RandomForestClassifier(n_estimators=400, max_depth=None,
  min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
  random_state=42)`, chosen with `RandomizedSearchCV` (20 combinations, macro
  F1).
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the Kaggle CSV to `data/wine_quality/winequalityN.csv`.
2. Open `training/wine/notebook.ipynb` with the `.venv` kernel and run every
   cell (about 1-2 minutes).
3. It generates `artifacts/wine.joblib` (about 50 MB, git-ignored) and
   `artifacts/wine.meta.json`.

## Evaluation (test set, 1066 wines)

| Metric | Model | Baseline (most frequent class) |
|---|---|---|
| Macro F1 | **0.5897** | 0.2028 |
| Accuracy | 0.5910 | - |
| Macro precision | 0.5801 | - |
| Macro recall | 0.6118 | - |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| high | 0.51 | 0.67 | 0.58 | 202 |
| low | 0.66 | 0.70 | 0.68 | 398 |
| medium | 0.58 | 0.46 | 0.51 | 466 |

Most important features: alcohol, density, volatile acidity and chlorides.
Errors are almost always between neighboring classes; performance is lower on
red wines (macro F1 0.54) than on white wines (0.60).

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
  Modeling wine preferences by data mining from physicochemical properties.
  *Decision Support Systems, 47*(4), 547-553.
  https://doi.org/10.1016/j.dss.2009.05.016
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
  https://doi.org/10.1023/A:1010933404324
