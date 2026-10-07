# Body fat (`body_fat`)

## Problem

- Task: regression of a man's body fat percentage (% of body weight)
- What it answers: a quick estimate of body fat from two measurements taken
  at home (a tape measure and a scale), to follow a training or diet plan.
  It is an estimate, not a medical assessment.
- Voice commands: "JarvisTEC, body fat", "JarvisTEC, body fat percentage",
  "JarvisTEC, how much body fat do I have", "JarvisTEC, fat percentage"

## Dataset

- Source: Kaggle
  [fedesoriano/body-fat-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset)
  (`bodyfat.csv`), the measurements of 252 men published by Johnson (1996)
  from the data of Penrose et al. (1985).
- License: check the license shown on the Kaggle page before
  redistributing.
- Expected file: `data/body_fat/bodyfat.csv`
- 252 rows x 15 columns; no missing values, no duplicated rows.
- Target: `body_fat` (column `BodyFat`), computed from the body density
  with the Siri equation (`495 / density - 450`).
- `density` is excluded: it gives the target directly (correlation -0.988;
  R2 0.99996 on the cleaned data) and needs underwater weighing.
- 6 rows removed (246 remain): 5 whose body fat contradicts their own
  density by more than 1 point (indices 47, 75, 95, 168 and 181, the last
  one with a body fat of 0) and 1 with an impossible height of 29.5 inches
  (index 41). The 164.7 kg man (index 38) is kept: his measurements are
  coherent.
- Units: the file has weight in pounds and height in inches; the notebook
  converts them to kg and cm. Circumferences are already in cm. Columns
  are renamed to snake_case with the unit as a suffix (`Abdomen` becomes
  `abdomen_cm`); they are the API input fields.

## Configuration

- `SEED = 42`, 80/20 split (196 train / 50 test), 5-fold cross-validation
  repeated 5 times, on the training set only.
- Inputs: `abdomen_cm` (abdomen circumference, cm) and `weight_kg` (body
  weight, kg), chosen by forward selection among the 13 measurements: the
  smallest set within 0.1 points of the best cross-validated RMSE (4.55
  with 2 inputs, 4.45 with the best 4, 4.55 with all 13).
- Preprocessing inside the `Pipeline`: median imputation +
  `StandardScaler`.
- Model: `Ridge(alpha=0.01)`, chosen over Lasso, Random Forest and Gradient
  Boosting by cross-validated RMSE; `alpha` searched over 16 values from
  0.01 to 1000 (the result is practically ordinary least squares).
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`;
  load the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/body_fat/bodyfat.csv`.
2. Open `training/body_fat/notebook.ipynb` with the `.venv` kernel and run
   every cell (about a minute).
3. It generates `artifacts/body_fat.joblib` (under 1 MB, committed) and
   `artifacts/body_fat.meta.json`.

## Evaluation (test set, 50 men)

| Metric | Model | Baseline (mean body fat) |
|---|---|---|
| RMSE (points) | **4.2676** | 7.4509 |
| MAE (points) | 3.2275 | - |
| R2 | 0.6716 | - |
| Within 3 / 5 points | 56 % / 82 % | - |

| Body fat | Men | MAE (points) | Mean error (points) |
|---|---|---|---|
| < 10 % | 5 | 2.56 | +2.56 |
| 10-20 % | 19 | 3.27 | +1.26 |
| 20-30 % | 22 | 3.01 | -1.47 |
| > 30 % | 4 | 5.03 | +1.62 |

The fitted model is
`body_fat = -48.109 + 1.0055 * abdomen_cm - 0.3173 * weight_kg`. The test
set is small: the bootstrap 95 % interval of its RMSE is 3.25 to 5.25. The
largest error (+13.3 points) is the 164.7 kg man, far outside the training
range (abdomen 70.4-126.2 cm, weight 56.7-119.18 kg); the backend flags
such inputs with `outside_training_range` and clips the estimate to the
body fat range seen in training (0.7-47.5 %). The data contains only adult
men, so the model must not be used for women or children.

## References

- Johnson, R. W. (1996). Fitting percentage of body fat to simple body
  measurements. *Journal of Statistics Education, 4*(1).
- Penrose, K. W., Nelson, A. G., & Fisher, A. G. (1985). Generalized body
  composition prediction equation for men using simple measurement
  techniques. *Medicine and Science in Sports and Exercise, 17*(2), 189.
- Siri, W. E. (1956). The gross composition of the body. *Advances in
  Biological and Medical Physics, 4*, 239-280.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased
  estimation for nonorthogonal problems. *Technometrics, 12*(1), 55-67.
