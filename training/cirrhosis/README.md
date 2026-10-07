# Cirrhosis stage (`cirrhosis`)

## Problem

- Task: multiclass classification of the histologic stage of liver disease
  (`stage`, 1 to 4, ordered) in patients with primary biliary cirrhosis.
- What it answers: a rough estimate of how advanced the disease is, from
  age, sex, four physical findings and nine lab values. **It is an estimate,
  not a diagnosis**; the stage is established with a biopsy.
- Voice commands: "JarvisTEC, cirrhosis stage", "JarvisTEC, stage of
  cirrhosis", "JarvisTEC, type of cirrhosis"

## Dataset

- Source: Kaggle
  [fedesoriano/cirrhosis-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)
  (`cirrhosis.csv`), the Mayo Clinic primary biliary cirrhosis trial data
  (Fleming & Harrington, 1991).
- License: check the license shown on the Kaggle page before
  redistributing.
- Expected file: `data/cirrhosis/cirrhosis.csv`
- 418 rows x 20 columns; 312 patients of the randomized trial and 106
  outside it. 6 rows have no `stage` (removed: 412 remain). No duplicates.
- Missing values: the 100 remaining patients outside the trial lack 8 of the
  15 inputs (`ascites`, `hepatomegaly`, `spiders`, `cholesterol`, `copper`,
  `alk_phos`, `sgot`, `triglycerides`); imputed inside the `Pipeline`.
- Target: `stage` (37.6 % stage 3, 35.0 % stage 4, 22.3 % stage 2, 5.1 %
  stage 1).
- Not inputs: `id` (identifier); `n_days` and `status` (follow-up time and
  outcome, only known after the stage was assessed: leakage); `drug`
  (assigned at random, independent of the stage: chi-square p = 0.20).
- Columns are renamed to lowercase (`Tryglicerides` becomes
  `triglycerides`) and `Age` is converted from days to years; they are the
  API input fields. The CSV carries no units; the laboratory units used in
  the schema and the notebook are an assumption.

## Configuration

- `SEED = 42`, stratified 80/20 split (329 train / 83 test), stratified
  5-fold cross-validation repeated 3 times.
- Preprocessing inside the `Pipeline`: median + `StandardScaler` (numeric),
  mode + `OneHotEncoder` (`sex`, `ascites`, `hepatomegaly`, `spiders`,
  `edema`).
- Model: `RandomForestClassifier(n_estimators=400, max_depth=8,
  min_samples_leaf=5, max_features="sqrt",
  class_weight="balanced_subsample", random_state=42)`, chosen with
  `RandomizedSearchCV` (20 combinations, macro F1).
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/cirrhosis/cirrhosis.csv`.
2. Open `training/cirrhosis/notebook.ipynb` with the `.venv` kernel and run
   every cell (a few minutes).
3. It generates `artifacts/cirrhosis.joblib` (about 1 MB, committed) and
   `artifacts/cirrhosis.meta.json`.

## Evaluation (test set, 83 patients)

| Metric | Model | Baseline (always stage 3) |
|---|---|---|
| Macro F1 | **0.4430** | 0.1360 |
| Balanced accuracy | 0.4450 | 0.2500 |
| Accuracy | 0.5060 | 0.3735 |
| Within one stage | 0.8916 | 0.9518 |
| Mean absolute error (stages) | 0.6024 | 0.6747 |
| Quadratic weighted kappa | 0.4856 | 0.0000 |

| Stage | Precision | Recall | F1 | Patients |
|---|---|---|---|---|
| 1 | 0.25 | 0.25 | 0.25 | 4 |
| 2 | 0.33 | 0.42 | 0.37 | 19 |
| 3 | 0.48 | 0.42 | 0.45 | 31 |
| 4 | 0.71 | 0.69 | 0.70 | 29 |

The result is modest: the model gives the exact stage for half of the
patients. It recognizes stage 4 reasonably well and mostly confuses stages 2
and 3 with each other; 42 predictions are exact, 32 are one stage away, 9
are two stages away and none is three stages away. The baseline is within
one stage more often only because stage 3 is next to every stage except 1;
it never identifies stage 4. In the test set 4 of the 29 stage 4 patients
were predicted as stage 2. The most used variables are platelets, albumin,
age, bilirubin and prothrombin time. The test set is small (4 patients of
stage 1), so every figure has a wide margin of uncertainty.

## References

- Fleming, T. R., & Harrington, D. P. (1991). *Counting processes and
  survival analysis*. Wiley.
- Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D., &
  Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: Model for
  decision making. *Hepatology, 10*(1), 1-7.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
  https://doi.org/10.1023/A:1010933404324
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE
  Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284.
  https://doi.org/10.1109/TKDE.2008.239
