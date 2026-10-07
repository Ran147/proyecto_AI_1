# Avocado price (`avocado`)

## Problem

- Task: time-series forecast of the average retail price of a Hass avocado
  for the **next week**, in USD per unit (United States).
- What it answers: a one-week-ahead reference price for a buyer or retailer
  planning purchases. The data ends on 2018-03-25, so the default answer is
  the forecast for the week of 2018-04-01, not a live price.
- Scope: trained on all 108 weekly series (54 regions x 2 types, lags
  computed inside each series); served for the national series (`TotalUS`),
  conventional or organic.
- Voice commands: "JarvisTEC, avocado price", "JarvisTEC, price of
  avocado", "JarvisTEC, price of avocados", "JarvisTEC, avocado
  forecast", "JarvisTEC, how much will avocado cost", "JarvisTEC,
  avocado". The whole word "organic" or "conventional" in the command
  selects the type ("JarvisTEC, organic avocado price").

## Dataset

- Source: Kaggle
  [neuromusic/avocado-prices](https://www.kaggle.com/datasets/neuromusic/avocado-prices),
  weekly retail scan data published by the Hass Avocado Board.
- License: check the license shown on the Kaggle page before
  redistributing.
- Expected file: `data/avocado/avocado.csv`
- 18,249 rows x 13 columns (plus a row counter that is dropped); 169 weeks
  from 2015-01-04 to 2018-03-25; no missing values, no duplicates. One
  series (`WestTexNewMexico` organic) lacks 3 weeks.
- Target: `average_price` (`AveragePrice` in the file).
- The volume and bag columns are same-week sales figures, unknown when the
  forecast is made: they are never inputs. `region` is not an input either.

## Configuration

- `SEED = 42`. Chronological split by week: 132 weeks for training
  (2015-02-01 to 2017-08-06, 14,245 rows) and the most recent 33 weeks for
  testing (2017-08-13 to 2018-03-25, 3,564 rows). Cross-validation:
  `TimeSeriesSplit` with 5 folds over the training weeks. Nothing is
  shuffled.
- Features (function `build_features`, identical in the notebook and in
  `backend/models/avocado/adapter.py`): `lag_1` to `lag_4` (prices of the 4
  previous weeks), `roll_mean_4` (their mean), `diff_1` (last change),
  `week_sin` and `week_cos` (ISO week of the forecast date, period 52) and
  `type`.
- Preprocessing inside the `Pipeline`: `StandardScaler` (numeric) and
  `OneHotEncoder` (`type`).
- Model: `Ridge(alpha=0.01)`, chosen by cross-validated RMSE against Random
  Forest, Gradient Boosting and Hist Gradient Boosting; `alpha` chosen with
  `RandomizedSearchCV` (no measurable effect). The exported pipeline is
  refitted on all 165 weeks.
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download the CSV to `data/avocado/avocado.csv`.
2. Open `training/avocado/notebook.ipynb` with the `.venv` kernel and run
   every cell (about a minute).
3. It generates `artifacts/avocado.joblib` (about 3 KB, committed) and
   `artifacts/avocado.meta.json`. The backend needs **both**: `meta.json`
   holds `history` (the last 8 national prices of each type and the date of
   the last observation), from which the adapter builds the lags, and
   `expected_default_prediction`, checked by
   `tests/models/test_avocado.py`.

## Serving

- No inputs: forecasts the week after the last date of the stored history
  for `type = "conventional"`.
- Optional inputs: `type` (`conventional` / `organic`); `recent_prices`, the
  user's own latest weekly prices (at least 4, most recent last, USD per
  avocado, each above 0 and at most 10); `last_date`, the date of the
  most recent of those prices (between 2000-01-01 and 2100-01-01;
  anything else is rejected with HTTP 422).
- `recent_prices` without `last_date`: the forecast is for the week after
  the user's latest price. `forecast_date` and `last_data_date` are
  `null` in the result and the spoken sentence names no calendar date;
  the week-of-year features assume the week after the stored history.
- The spoken sentence (English) gives the price in US dollars per unit,
  the type, the forecast week and the date of the last data point (when
  known), and states that it is a forecast on historical data ending in
  March 2018, not a current price.
- Default forecast stored in `meta.json`: 1.0523 USD (conventional, last
  price 1.03) and 1.5496 USD (organic, last price 1.55) for the week of
  2018-04-01.

## Evaluation (test set, 33 weeks, one-week-ahead forecasts)

| Test set | Forecast | RMSE (USD) | MAE (USD) | MAPE | R2 |
|---|---|---|---|---|---|
| All 108 series (3,564) | **Model (Ridge)** | **0.1531** | 0.1114 | 7.75 % | 0.8501 |
| | Naive (last week) | 0.1636 | 0.1124 | 7.98 % | 0.8288 |
| | Seasonal naive (52 weeks ago) | 0.4176 | 0.3212 | 20.24 % | -0.1161 |
| `TotalUS` (66) | Model (Ridge) | 0.0794 | 0.0618 | 4.57 % | 0.9383 |
| | **Naive (last week)** | **0.0764** | 0.0591 | 4.49 % | 0.9429 |

- Over all the series the model beats the naive forecast "next week = this
  week" by 6.4 % in RMSE (72 of the 108 series), mostly by reducing the
  error after large jumps.
- **On the national series, the one that is served, it does not beat the
  naive forecast** (RMSE 3.9 % higher; a difference of 0.003 USD over 66
  forecasts). The national weekly price is close to a random walk at this
  horizon.
- The fitted model is a smoothed naive forecast: weights 0.57, 0.14, 0.08
  and 0.12 on the four previous weeks (sum 0.91).

## References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
  practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical
  and machine learning forecasting methods: Concerns and ways forward.
  *PLoS ONE, 13*(3), e0194889. https://doi.org/10.1371/journal.pone.0194889
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation
  for nonorthogonal problems. *Technometrics, 12*(1), 55-67.
