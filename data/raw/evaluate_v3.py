"""
evaluate_v3.py
==============
Corrected 1-step-ahead evaluation using predict(dynamic=False).

Root cause of ARMAX(2,1,2) apply() failure:
  - fitted.apply(endog_test, ...) with d=1 models restarts the state from
    a non-stationary initialization, causing prediction instability.

Correct method:
  - Refit the model on the FULL dataset (train+test) using trained params
    as starting values, then call:
        result.predict(start=n_train, end=n_total-1, dynamic=False)
  - dynamic=False means at each step t, it uses ACTUAL y(t-1) instead
    of its previous predictions -> true 1-step-ahead conditional forecast.
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
DATA_DIR   = r"d:\Downloads\Time Series\TIME SERIES FINAL\Cleaned"
OUTPUT_CSV = r"d:\Downloads\Time Series\TIME SERIES FINAL\evaluate_v3_results.csv"

GAMES = {
    "EldenRing"   : "EldenRing_Cleaned.csv",
    "HuntShowdown": "HuntShowdown_Cleaned.csv",
    "RDR2"        : "RDR2_Cleaned.csv",
    "ReadyorNot"  : "ReadyorNot_Cleaned.csv",
}
BEST_ORDERS = {
    "EldenRing"   : (2, 1, 2),
    "HuntShowdown": (1, 1, 1),
    "RDR2"        : (2, 1, 2),
    "ReadyorNot"  : (2, 1, 2),
}
TRAIN_RATIO = 0.80
MAX_ITER    = 300
# ---------------------------------------------------------------------------

def prepare_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq("D")
    df["Avg_Player"] = df["Avg_Player"].interpolate("linear")
    df["Log_Player"] = np.log(df["Avg_Player"] + 1)
    df["Discount_Ratio"]      = df["Discount_%"] / 100.0
    df["Years_Since_Release"] = df["Days_Since_Release"] / 365.0
    for col in ["Discount_Ratio", "Years_Since_Release", "Is_Weekend", "Is_Holiday"]:
        df[col] = df[col].ffill().bfill()
    endog = df["Log_Player"]
    exog  = df[["Years_Since_Release", "Discount_Ratio", "Is_Weekend", "Is_Holiday"]]
    return df, endog, exog


def mape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(mape_score(np.array(y_true), np.array(y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, mape, r2


def seasonal_naive(endog_train, endog_test, s=7):
    full = pd.concat([endog_train, endog_test])
    preds = []
    for idx in endog_test.index:
        lag = idx - pd.Timedelta(days=s)
        preds.append(full.loc[lag] if lag in full.index else endog_train.iloc[-1])
    return np.array(preds)


def onestep_predict(endog, exog, n_train, p, d, q):
    """
    Fit on train, then predict(dynamic=False) from n_train onward.
    dynamic=False = at each step, use ACTUAL y(t-1), not model's previous forecast.
    This is the correct 1-step-ahead evaluation.
    """
    endog_train = endog.iloc[:n_train]
    exog_train  = exog.iloc[:n_train]

    # Step 1: Fit on train to get params
    m = ARIMA(endog=endog_train, exog=exog_train, order=(p, d, q),
              enforce_stationarity=True, enforce_invertibility=True)
    fitted = m.fit(method="statespace", method_kwargs={"maxiter": MAX_ITER})

    # Step 2: Refit full model (train+test) starting from trained params
    # Use start_params to pass trained model params -> no re-estimation
    m_full = ARIMA(endog=endog, exog=exog, order=(p, d, q),
                   enforce_stationarity=True, enforce_invertibility=True)
    fitted_full = m_full.fit(
        start_params=fitted.params,
        method="statespace",
        method_kwargs={"maxiter": 1}   # near-instant: just 1 iter from good start
    )

    # Step 3: predict with dynamic=False on test portion
    n_total = len(endog)
    preds = fitted_full.predict(
        start=n_train,
        end=n_total - 1,
        exog=exog.iloc[n_train:],
        dynamic=False    # KEY: use actual y at each step, not chained preds
    )
    return fitted, preds


def main():
    all_rows = []

    print()
    print("=" * 80)
    print("  CORRECTED 1-STEP-AHEAD EVALUATION  [predict(dynamic=False)]")
    print("=" * 80)
    print()
    print("  dynamic=False -> at each step t, uses ACTUAL y(t-1), not model forecast")
    print()

    for game_name, csv_file in GAMES.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        p, d, q  = BEST_ORDERS[game_name]

        print("-" * 80)
        print(f"  GAME: {game_name}  |  Best model: ARMAX({p},{d},{q})")
        print("-" * 80)

        df, endog, exog = prepare_data(csv_path)
        n_total = len(endog)
        n_train = int(n_total * TRAIN_RATIO)
        n_test  = n_total - n_train
        endog_test = endog.iloc[n_train:]

        print(f"  Test: {n_test} rows | {endog_test.index[0].date()} -> {endog_test.index[-1].date()}")
        print()

        # Baseline: Seasonal Naive
        naive = seasonal_naive(endog.iloc[:n_train], endog_test, s=7)
        rm, ma, mp, r2 = metrics(endog_test.values, naive)
        print(f"    Baseline Seasonal Naive:          RMSE={rm:.4f}  MAE={ma:.4f}  MAPE={mp:.2f}%  R2={r2:.4f}")
        all_rows.append(dict(Game=game_name, Model="SeasonalNaive(7)",
                             RMSE=round(rm,5), MAE=round(ma,5), MAPE=round(mp,3), R2=round(r2,4)))

        # ARMAX(1,0,0) — notebook baseline
        print(f"    Fitting ARMAX(1,0,0) ...", end="", flush=True)
        try:
            _, pred100 = onestep_predict(endog, exog, n_train, 1, 0, 0)
            print(" done.")
            rm, ma, mp, r2 = metrics(endog_test.values, pred100.values)
            print(f"    ARMAX(1,0,0) 1-step [dynamic=F]:  RMSE={rm:.4f}  MAE={ma:.4f}  MAPE={mp:.2f}%  R2={r2:.4f}")
            all_rows.append(dict(Game=game_name, Model="ARMAX(1,0,0)_dynF",
                                 RMSE=round(rm,5), MAE=round(ma,5), MAPE=round(mp,3), R2=round(r2,4)))
        except Exception as e:
            print(f" ERROR: {e}")

        # Best model
        print(f"    Fitting ARMAX({p},{d},{q}) ...", end="", flush=True)
        try:
            fitted_best, pred_best = onestep_predict(endog, exog, n_train, p, d, q)
            print(" done.")
            rm, ma, mp, r2 = metrics(endog_test.values, pred_best.values)
            print(f"    ARMAX({p},{d},{q}) 1-step [dynamic=F]: RMSE={rm:.4f}  MAE={ma:.4f}  MAPE={mp:.2f}%  R2={r2:.4f}  <- CORRECT")

            # Ljung-Box on 1-step residuals
            resid1 = endog_test.values - pred_best.values
            lb7  = acorr_ljungbox(resid1, lags=[7],  return_df=True)["lb_pvalue"].values[0]
            lb14 = acorr_ljungbox(resid1, lags=[14], return_df=True)["lb_pvalue"].values[0]
            print(f"    Ljung-Box: lag7={lb7:.4f} ({'OK' if lb7>0.05 else 'FAIL'})  "
                  f"lag14={lb14:.4f} ({'OK' if lb14>0.05 else 'FAIL'})")

            all_rows.append(dict(Game=game_name, Model=f"ARMAX({p},{d},{q})_dynF",
                                 RMSE=round(rm,5), MAE=round(ma,5), MAPE=round(mp,3), R2=round(r2,4)))
        except Exception as e:
            print(f" ERROR: {e}")

        print()

    # Final summary
    df_out = pd.DataFrame(all_rows)[["Game","Model","RMSE","MAE","MAPE","R2"]]
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("=" * 80)
    print("  FINAL SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"  {'Game':<14} {'Model':<30} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'R2':>8}")
    print(f"  {'-'*72}")
    for _, row in df_out.iterrows():
        marker = "  ***" if "dynF" in row["Model"] and "1,0,0" not in row["Model"] else ""
        print(f"  {row['Game']:<14} {row['Model']:<30} "
              f"{row['RMSE']:>8.4f} {row['MAE']:>8.4f} {row['MAPE']:>7.2f} {row['R2']:>8.4f}{marker}")
    print()
    print(f"  *** = Recommended model with correct evaluation")
    print(f"  Results -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
