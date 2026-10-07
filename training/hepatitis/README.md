# Hepatitis C category (`hepatitis`)

## Problem

- Task: multiclass classification of a patient's category: `blood_donor`,
  `hepatitis`, `fibrosis` or `cirrhosis` (hepatitis C at increasing stages
  of liver damage)
- What it answers: an estimate of the group a routine blood panel (10 lab
  values, age and sex) looks like, to help decide which samples deserve a
  closer look. It is an estimate, **not a diagnosis**.
- Voice commands: "JarvisTEC, hepatitis type", "JarvisTEC, type of
  hepatitis", "JarvisTEC, classify hepatitis"

## Dataset

- Source: Kaggle
  [fedesoriano/hepatitis-c-dataset](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset)
  (`HepatitisCdata.csv`), a copy of the *HCV data* set of the UCI Machine
  Learning Repository (Lichtinghagen, Klawonn & Hoffmann).
- License: not verified for this project; check the license shown on the
  Kaggle page and on the UCI page before redistributing the data.
- Expected file: `data/hepatitis/HepatitisCdata.csv`
- 615 rows x 14 columns; the first column is a row number and is dropped.
  31 missing values in 5 columns (`alp` 18, `chol` 10, `alb`, `alt` and
  `prot` 1 each); no duplicated rows.
- Target: `Category`, renamed to `category` with clean labels. Original
  counts: 533 blood donors, 24 hepatitis, 21 fibrosis, 30 cirrhosis and 7
  "suspect" blood donors.
- **The 7 suspect donors are removed** (608 patients remain): their label is
  not a confirmed state, 7 rows cannot be learned or evaluated, and their
  lab values are atypical for donors. The notebook only shows what the model
  answers for them.
- Columns are renamed to lowercase (`AST` becomes `ast`); they are the API
  input fields. The source does not document the units; the schema uses the
  usual reporting units (g/L, U/L, umol/L, kU/L, mmol/L), which are
  consistent with the value ranges. **These units are an assumption** and
  have not been confirmed against a primary source.

## Configuration

- `SEED = 42`, stratified 80/20 split (486 train / 122 test), stratified
  5-fold cross-validation repeated 3 times, on the training set only.
- Preprocessing inside the `Pipeline`: median + `log(1 + x)` +
  `StandardScaler` (numeric), mode + `OneHotEncoder` (`sex`).
- Model: `RandomForestClassifier(n_estimators=200, max_depth=None,
  min_samples_leaf=1, max_features="sqrt", class_weight="balanced",
  random_state=42)`. Chosen among 11 candidates by cross-validated macro F1
  (0.699); logistic regression (0.694) and an SVM (0.693) were tied with it.
  `RandomizedSearchCV` (20 combinations) did not improve it, so the initial
  configuration is kept.
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/hepatitis/HepatitisCdata.csv`.
2. Open `training/hepatitis/notebook.ipynb` with the `.venv` kernel and run
   every cell (about a minute).
3. It generates `artifacts/hepatitis.joblib` (about 1.3 MB, committed) and
   `artifacts/hepatitis.meta.json`.

## Evaluation (test set, 122 patients)

| Metric | Model | Baseline (most frequent class) |
|---|---|---|
| Macro F1 | **0.5879** | 0.2336 |
| Balanced accuracy | 0.5893 | 0.25 |
| Accuracy | 0.9262 | 0.877 |
| Macro precision | 0.5966 | - |
| Macro recall | 0.5893 | - |

| Class | Precision | Recall | F1 | Test patients |
|---|---|---|---|---|
| blood_donor | 0.98 | 0.99 | 0.99 | 107 |
| hepatitis | 0.33 | 0.20 | 0.25 | 5 |
| fibrosis | 0.50 | 0.50 | 0.50 | 4 |
| cirrhosis | 0.57 | 0.67 | 0.62 | 6 |

**Read these numbers with care.** The test set has only 4 to 6 patients per
disease class: one patient changes a class recall by 17 to 25 points. The
cross-validated macro F1 on the training set was 0.699 (standard deviation
0.105 across folds).

The model separates donors from patients well (13 of the 15 test patients
with disease were assigned to some disease class and 1 of 107 donors was
flagged) but it does not stage the disease reliably: only 7 of those 13
patients received the right stage. AST is by far the most important input,
followed by ALP and cholinesterase. The default input of the backend is the
median training patient, a healthy profile, so a voice command without data
answers "blood donor".

## References

- Hoffmann, G., Bietenbeck, A., Lichtinghagen, R., & Klawonn, F. (2018).
  Using machine learning techniques to generate laboratory diagnostic
  pathways—a case study. *Journal of Laboratory and Precision Medicine, 3*,
  58.
- Lichtinghagen, R., Klawonn, F., & Hoffmann, G. (2020). *HCV data*
  [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5D612
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
  https://doi.org/10.1023/A:1010933404324
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE
  Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284.
  https://doi.org/10.1109/TKDE.2008.239
