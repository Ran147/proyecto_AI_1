# Bitcoin price for tomorrow (`bitcoin`)

## Problem

- Task: time-series forecast of the next-day Bitcoin closing price, in USD.
- What it answers: an estimate of tomorrow's close from the closes of the
  previous 31 days, and how much (how little) can be known about it.
- Honest result: the model **does not beat** the naive forecast "tomorrow =
  today" (see Evaluation). Read its answer as "about the same as today, give
  or take 2 to 3 %", never as a trading signal.
- The data ends on 2017-07-31, so the default forecast is for 2017-08-01. It
  is not a live price, and `speak()` says the dates.
- Voice commands: "JarvisTEC, bitcoin price", "JarvisTEC, price of
  bitcoin", "JarvisTEC, bitcoin price tomorrow", "JarvisTEC, bit coin",
  "JarvisTEC, bitcoin"

## Dataset

- Source: Kaggle
  [team-ai/bitcoin-price-prediction](https://www.kaggle.com/datasets/team-ai/bitcoin-price-prediction).
- License: check the license shown on the Kaggle page before redistributing.
- Expected files:
  - `data/bitcoin/bitcoin_price_Training - Training.csv`: 1556 rows x 7
    columns, one per day from 2013-04-28 to 2017-07-31.
  - `data/bitcoin/bitcoin_price_1week_Test - Test.csv`: the next 7 days
    (2017-08-01 to 2017-08-07), used only as an extra check.
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Market Cap`,
  renamed to snake_case. Dates are text (`"Jul 31, 2017"`) and rows are
  newest-first: they are parsed and sorted ascending. `Volume` and `Market
  Cap` are text with thousands separators; `Volume` is `-` (missing) on 243
  days, all before 2013-12-27.
- No calendar gaps, no duplicated dates, no missing closes.
- Target: `close` of the next day. Only past closes are inputs: the same-day
  `open`, `high`, `low`, `volume` and `market_cap` are not known at forecast
  time.

## Configuration

- `SEED = 42`. Chronological split of the 1525 forecast days: first 80 %
  train (1220 days, 2013-05-29 to 2016-09-29), last 20 % test (305 days,
  2016-09-30 to 2017-07-31). Cross-validation: `TimeSeriesSplit(5)`. Nothing
  is shuffled.
- Features (`build_features`, from the 31 closes before the forecast day): 7
  lagged log returns, mean and standard deviation of the returns over 7 and
  30 days, log distance of the last close to its 7- and 30-day moving
  averages, and the day of the week of the forecast day. The 7 price lags
  are also built, for the price-level candidates.
- The model predicts the next-day **log return**; the price is
  `last close * exp(return)`. Modelling the price level directly fails with
  trees (they cannot extrapolate): Random Forest on price lags gets a test
  RMSE of 800.57 USD.
- Model: `RandomForestRegressor(n_estimators=200, max_depth=6,
  min_samples_leaf=50, max_features=0.33, random_state=42)` inside a
  `Pipeline` (`StandardScaler` on the numeric features, `OneHotEncoder` on
  the day of the week). Chosen as the best non-baseline candidate by
  cross-validated RMSE in USD and tuned over an 18-combination grid, on the
  training days only.
- The exported pipeline is refitted on all 1525 days with those
  hyperparameters; the test metrics below come from the train-only fit.
- Trained with scikit-learn 1.9.1 (see `sklearn_version` in `meta.json`; load
  the `.joblib` with the same version).

## Reproduce

1. Download both CSV files to `data/bitcoin/`.
2. Open `training/bitcoin/notebook.ipynb` with the `.venv` kernel and run
   every cell (about a minute).
3. It generates `artifacts/bitcoin.joblib` (under 1 MB, committed) and
   `artifacts/bitcoin.meta.json`. The meta file holds the `history` the
   backend forecasts from (last 31 closes, `last_date`, `forecast_date`) and
   `expected_default_prediction` (2889.08 USD for 2017-08-01).
4. If the data changes, copy `default_input` (the last close) from
   `meta.json` to `backend/models/bitcoin/adapter.py`;
   `tests/models/test_bitcoin.py` checks that they match.

## Serving (`backend/models/bitcoin/`)

- `build_features` in `adapter.py` is a copy of the notebook function; a test
  checks that the adapter reproduces `expected_default_prediction`.
- No inputs (or `last_price` equal to the last close of the data, 2875.34):
  forecast for the day after the data ends.
- `last_price`: the user's latest close. The stored 31-day history is
  rescaled to that price level (same recent day-to-day pattern), so the
  forecast is the same expected daily change applied to the user's price.
  Replacing only the last stored value would turn a price far from 2875 USD
  into a one-day jump the model never saw.
- `recent_prices`: latest daily closes, most recent last; it takes precedence
  over `last_price`. They replace the end of the history and the older stored
  closes are rescaled to join them; 31 or more values replace it entirely.
- With user prices the forecast is "for the day after that price" and has no
  date; the day-of-week feature keeps the stored forecast day.
- `last_price` and every item of `recent_prices` must be finite, above 0
  and at most 10,000,000 USD; anything else is rejected with HTTP 422.

## Evaluation (test set, 305 days, 2016-09-30 to 2017-07-31)

| Metric | Model | Naive (tomorrow = today) | Drift (mean return) |
|---|---|---|---|
| RMSE (USD) | 69.9825 | **69.7329** | 69.5680 |
| MAE (USD) | 39.1822 | 40.0305 | 39.6741 |
| MAPE | 2.4405 % | 2.4977 % | 2.4702 % |
| R2 | 0.9903 | 0.9903 | 0.9904 |
| Directional accuracy | 0.6250 | not defined | 0.6382 |

- The model does **not** beat the naive forecast on RMSE (ratio 1.004). It is
  slightly better on MAE and MAPE, but the difference is not statistically
  significant (paired t-test on the absolute percentage errors: t = 1.79,
  p = 0.074).
- Directional accuracy is no better than always answering "up": 63.6 % of the
  test days closed higher.
- R2 on the price level is not evidence of skill: the naive forecast has the
  same value.
- Cross-validation on the training days told the same story: naive 15.50 USD,
  tuned model 15.64 USD.

Extra check on the 7-day Kaggle file (2017-08-01 to 2017-08-07, one-day-ahead
forecasts with the exported model):

| Metric | Model | Naive |
|---|---|---|
| RMSE (USD) | 167.3946 | 168.2244 |
| MAE (USD) | 132.5809 | 130.1257 |
| MAPE | 4.3160 % | 4.2331 % |
| Directional accuracy | 0.5714 (4 of 7) | not defined |

Seven days are too few to conclude anything; it is again a tie.

Limitations: no advantage over the naive forecast; data ends in July 2017;
only past closes are used (no volume, news or sentiment); a single point
forecast with no uncertainty range; one asset and one chronological split.

## References

- Nakamoto, S. (2008). *Bitcoin: A peer-to-peer electronic cash system*.
  https://bitcoin.org/bitcoin.pdf
- McNally, S., Roche, J., & Caton, S. (2018). Predicting the price of Bitcoin
  using machine learning. In *2018 26th Euromicro International Conference on
  Parallel, Distributed and Network-based Processing (PDP)* (pp. 339-343).
  IEEE. https://doi.org/10.1109/PDP2018.2018.00060
- Fama, E. F. (1970). Efficient capital markets: A review of theory and
  empirical work. *The Journal of Finance, 25*(2), 383-417.
  https://doi.org/10.2307/2325486
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
  practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
  https://doi.org/10.1023/A:1010933404324
