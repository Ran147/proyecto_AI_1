# Customer churn (`churn`)

## Problem

- Task: binary classification of whether a telecom customer will leave the
  company (`Yes` / `No`). The model also returns the churn probability.
- What it answers: which customers are at risk, so a retention team can
  contact them before they leave. It is an estimate to rank customers, not a
  certainty about one person.
- Voice commands: "JarvisTEC, customer churn", "JarvisTEC, will the
  customer leave", "JarvisTEC, is the customer going to leave",
  "JarvisTEC, customer leaving", "JarvisTEC, churn"

## Dataset

- Source: GitHub
  [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
  (`data/Telco-Customer-Churn.csv`), the IBM sample dataset *Telco Customer
  Churn*.
- License: not verified by the team; check the source repository before
  redistributing.
- Expected file: `data/churn/Telco-Customer-Churn.csv`
- 7043 rows x 21 columns, one customer per row; 26.5 % churn (1869 of 7043).
- `TotalCharges` is text with 11 blank cells, all customers with `tenure` 0:
  it is converted to a number and the blanks are set to 0. There are no other
  missing values and no repeated rows.
- Target: `churn` (`Yes` / `No`, trained as 1 / 0). `customer_id` is not an
  input.
- Columns are renamed to snake_case (`MonthlyCharges` becomes
  `monthly_charges`); the 19 remaining columns are the API input fields. The
  source does not state the currency of the charges (US dollars assumed).

## Configuration

- `SEED = 42`, stratified 80/20 split (5634 train / 1409 test), stratified
  5-fold cross-validation.
- Preprocessing inside the `Pipeline`: median + `StandardScaler` (4 numeric),
  mode + `OneHotEncoder` (15 categorical).
- Model: `GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,
  max_depth=2, min_samples_leaf=20, subsample=1.0, random_state=42)` wrapped
  in `TunedThresholdClassifierCV(scoring="f1", cv=5)`, chosen among 9
  candidates and tuned with `RandomizedSearchCV` (12 combinations, F1 of the
  churn class). Decision threshold: **0.299** instead of 0.5.
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/churn/Telco-Customer-Churn.csv`.
2. Open `training/churn/notebook.ipynb` with the `.venv` kernel and run every
   cell (about 5 minutes; the nested cross-validation is the slow part).
3. It generates `artifacts/churn.joblib` (about 55 KB, committed) and
   `artifacts/churn.meta.json`.

## Evaluation (test set, 1409 customers, 374 churners)

| Metric | Model | Baseline (always "stays") |
|---|---|---|
| F1, churn class | **0.6275** | 0.0 |
| Macro F1 | 0.7237 | 0.4235 |
| Macro precision | 0.7150 | - |
| Macro recall | 0.7614 | - |
| Precision, churn class | 0.5294 | - |
| Recall, churn class | 0.7701 | 0.0 |
| ROC-AUC | 0.8456 | - |
| Average precision | 0.6615 | - |
| Brier score | 0.1356 | - |
| Accuracy | 0.7573 | 0.7346 |

Confusion matrix (threshold 0.299):

| | Predicted stays | Predicted leaves |
|---|---|---|
| Stayed (1035) | 779 | 256 |
| Left (374) | 86 | 288 |

| Contract | Customers | Churn rate | Recall | Precision |
|---|---|---|---|---|
| Month-to-month | 773 | 0.426 | 0.869 | 0.534 |
| One year | 300 | 0.120 | 0.056 | 0.250 |
| Two year | 336 | 0.027 | 0.000 | 0.000 |

Segment results on the test set, by contract, are shown above. The model
finds about three out of four churners, but almost half of its alarms are
false, and it detects almost no churn in one-year contracts and none in
two-year contracts. Accuracy is only slightly above the baseline because the
threshold favors recall. Logistic regression performs the same within the
cross-validation (CV) noise (CV F1 0.633 vs. 0.636), and tuning raised the CV
F1 only from 0.6364 to 0.6386. On the test set the probabilities are well
calibrated on average (mean prediction 0.2663 vs. observed rate 0.2654). The
month-to-month contract (importance 0.36) and `tenure` (0.17) drive the
model; gender, partner, dependents and streaming services contribute
practically nothing (permutation importance of 0.0003 or less).

## References

- Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New
  insights into churn prediction in the telecommunication sector: A profit
  driven data mining approach. *European Journal of Operational Research,
  218*(1), 211-229. https://doi.org/10.1016/j.ejor.2011.09.031
- Huang, B., Kechadi, M. T., & Buckley, B. (2012). Customer churn prediction
  in telecommunications. *Expert Systems with Applications, 39*(1),
  1414-1425. https://doi.org/10.1016/j.eswa.2011.08.024
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
  machine. *The Annals of Statistics, 29*(5), 1189-1232.
  https://doi.org/10.1214/aos/1013203451
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE
  Transactions on Knowledge and Data Engineering, 21*(9), 1263-1284.
  https://doi.org/10.1109/TKDE.2008.239
