# Stroke risk (`stroke`)

## Problem

- Task: binary classification (`stroke` = 1 when the patient had a stroke).
  The model outputs a probability and a risk level (`low`, `moderate`,
  `high`).
- What it answers: how likely it is that a person with a given profile (age,
  hypertension, heart disease, glucose, BMI, smoking...) belongs to the group
  of patients who had a stroke. It is a screening estimate to decide who
  should get a medical check-up first, **not a diagnosis**.
- Voice commands: "JarvisTEC, stroke risk", "JarvisTEC, risk of a stroke",
  "JarvisTEC, risk of stroke", "JarvisTEC, stroke"

## Dataset

- Source: Kaggle
  [fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
  (`healthcare-dataset-stroke-data.csv`). The page does not document the
  hospital, country or period of the records.
- License: check the license and the terms of use shown on the Kaggle page
  before redistributing.
- Expected file: `data/stroke/healthcare-dataset-stroke-data.csv`
- 5110 rows x 12 columns; 201 missing values in `bmi` (written `N/A`); no
  duplicated rows; 1 row with gender `Other` (removed: 5109 remain).
- Target: `stroke` (249 positive patients, 4.87 %). `id` is not an input.
- Columns are renamed to lowercase (`Residence_type` becomes
  `residence_type`); they are the API input fields.

## Configuration

- `SEED = 42`, stratified 80/20 split (4087 train / 1022 test), stratified
  5-fold cross-validation.
- Preprocessing inside the `Pipeline`: median + `StandardScaler` (numeric,
  including the missing `bmi`), mode + `OneHotEncoder` (`gender`,
  `ever_married`, `work_type`, `residence_type`, `smoking_status`). No
  resampling and no "BMI missing" indicator.
- Model: `RandomForestClassifier(n_estimators=400, max_depth=10,
  min_samples_leaf=20, max_features=0.5, random_state=42)` without class
  weights, chosen among 6 candidates (including the baseline) by
  cross-validated ROC-AUC and tuned with `RandomizedSearchCV` (20
  combinations, ROC-AUC).
  Class weights were tried: they raise the recall at the 0.5 threshold but do
  not improve ROC-AUC and inflate the probabilities, so the imbalance is
  handled with the decision threshold instead.
- Decision threshold: **0.10** (best F2 on out-of-fold predictions of the
  training set). Risk levels: `low` below 0.10, `moderate` from 0.10 to 0.15,
  `high` from 0.15 (0.15 = best F1 on the same predictions). The test set was
  not used to choose them. They are stored in `meta.json` (`threshold`,
  `risk_thresholds`) and read by the backend adapter.
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/stroke/healthcare-dataset-stroke-data.csv`.
2. Open `training/stroke/notebook.ipynb` with the `.venv` kernel and run
   every cell (about a minute).
3. It generates `artifacts/stroke.joblib` (about 3 MB, committed) and
   `artifacts/stroke.meta.json`. The backend needs both files.

## Evaluation (test set, 1022 patients, 50 with a stroke)

| Metric | Model | Baseline (predicts the prevalence) |
|---|---|---|
| ROC-AUC | **0.8279** | 0.5000 |
| PR-AUC (average precision) | **0.2466** | 0.0489 |
| Recall, stroke class (threshold 0.10) | 0.7000 | 0.0000 |
| Precision, stroke class | 0.1832 | 0.0000 |
| F1, stroke class | 0.2905 | 0.0000 |
| Specificity | 0.8395 | 1.0000 |
| Accuracy (misleading here) | 0.8327 | 0.9511 |
| Brier score (lower is better) | 0.0417 | 0.0465 |

Confusion matrix at the threshold of 0.10:

| | Predicted no stroke | Predicted stroke |
|---|---|---|
| No stroke (972) | 816 | 156 |
| Stroke (50) | 15 | 35 |

| Risk level | Test patients | Strokes | Observed rate |
|---|---|---|---|
| low (< 0.10) | 831 | 15 | 1.8 % |
| moderate (0.10-0.15) | 81 | 9 | 11.1 % |
| high (>= 0.15) | 110 | 26 | 23.6 % |

The results are modest and must be read with care:

- Precision is low: 156 of the 191 flagged patients did not have a stroke.
- The model is driven by age. Age alone, used as a score, reaches a ROC-AUC
  of 0.8347 on the same test set (PR-AUC 0.2087): the model does not rank
  patients better than age; it only adds a probability and a better PR-AUC
  (0.2466).
- It flags nobody under 50: the 8 test patients under 50 with a stroke are
  all missed.
- With only 50 positive test patients every metric is uncertain (one patient
  changes the recall by 2 percentage points).
- The five candidates compared (logistic regression and Random Forest with
  and without class weights, Gradient Boosting) had a cross-validated ROC-AUC
  between 0.837 and 0.840, i.e. they are equivalent.

## References

- Feigin, V. L., et al. (2021). Global, regional, and national burden of
  stroke and its risk factors, 1990-2019: A systematic analysis for the
  Global Burden of Disease Study 2019. *The Lancet Neurology, 20*(10),
  795-820. https://doi.org/10.1016/S1474-4422(21)00252-0
- O'Donnell, M. J., et al. (2016). Global and regional effects of potentially
  modifiable risk factors associated with acute stroke in 32 countries
  (INTERSTROKE): A case-control study. *The Lancet, 388*(10046), 761-775.
  https://doi.org/10.1016/S0140-6736(16)30506-2
- Wolf, P. A., D'Agostino, R. B., Belanger, A. J., & Kannel, W. B. (1991).
  Probability of stroke: A risk profile from the Framingham Study. *Stroke,
  22*(3), 312-318. https://doi.org/10.1161/01.STR.22.3.312
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more
  informative than the ROC plot when evaluating binary classifiers on
  imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
  https://doi.org/10.1371/journal.pone.0118432
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
  https://doi.org/10.1023/A:1010933404324
