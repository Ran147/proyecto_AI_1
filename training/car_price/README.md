# Car price (`car_price`)

## Problem

- Task: regression of the resale price, in lakh of Indian rupees (1 lakh =
  100,000 INR)
- What it answers: a reference price for a used vehicle, for a seller who
  wants to publish it or a buyer who wants to check an asking price.
- Voice commands: "JarvisTEC, car price", "JarvisTEC, how much is my car
  worth", "JarvisTEC, used car price"

## Dataset

- Source: GitHub
  [amankharwal/Website-data](https://raw.githubusercontent.com/amankharwal/Website-data/master/car%20data.csv)
  (`car data.csv`), used-vehicle listings from the Indian marketplace
  CarDekho.
- License: not stated in the source repository; check before redistributing.
- Expected file: `data/car_price/car data.csv`
- 301 rows x 9 columns (cars and motorcycles); no missing values; 2
  duplicated rows (removed: 299 remain).
- Target: `selling_price`. `car_name` is not an input (98 distinct names).
- Columns are renamed to lowercase (`Present_Price` becomes
  `present_price`); they are the API input fields.

## Configuration

- `SEED = 42`, 80/20 split (239 train / 60 test), 5-fold cross-validation.
- Preprocessing inside the `Pipeline`: median + `StandardScaler` (numeric),
  mode + `OneHotEncoder` (`fuel_type`, `seller_type`, `transmission`).
- Model: `GradientBoostingRegressor(n_estimators=400, learning_rate=0.03,
  max_depth=2, subsample=0.8, random_state=42)` trained on `log(1 + price)`
  through `TransformedTargetRegressor`, chosen with `RandomizedSearchCV` (20
  combinations, RMSE in lakh).
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/car_price/car data.csv`.
2. Open `training/car_price/notebook.ipynb` with the `.venv` kernel and run
   every cell (under a minute).
3. It generates `artifacts/car_price.joblib` (under 1 MB, committed) and
   `artifacts/car_price.meta.json`.

## Evaluation (test set, 60 vehicles)

| Metric | Model | Baseline (mean price) |
|---|---|---|
| RMSE (lakh) | **1.7871** | 5.0768 |
| MAE (lakh) | 0.8724 | - |
| R2 | 0.8761 | - |
| Median % error | 12.3 % | - |

| Price segment | Vehicles | MAE (lakh) | Median % error |
|---|---|---|---|
| < 1 lakh | 14 | 0.07 | 18.1 % |
| 1-5 lakh | 24 | 0.78 | 11.8 % |
| 5-10 lakh | 17 | 0.91 | 7.3 % |
| > 10 lakh | 5 | 3.41 | 19.2 % |

The showroom price (`present_price`) holds 0.93 of the feature importance and
`year` 0.06. The model underestimates the most expensive vehicles and
overestimates old premium ones, both rare in the data.

## References

- Pudaruth, S. (2014). Predicting the price of used cars using machine
  learning techniques. *International Journal of Information & Computation
  Technology, 4*(7), 753-764.
- Gegic, E., Isakovic, B., Keco, D., Masetic, Z., & Kevric, J. (2019). Car
  price prediction using machine learning techniques. *TEM Journal, 8*(1),
  113-118. https://doi.org/10.18421/TEM81-16
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
  machine. *The Annals of Statistics, 29*(5), 1189-1232.
  https://doi.org/10.1214/aos/1013203451
