"""
model_selection.py
==================
Grid search over ARMAX(p, d, q) for each game in the Cleaned folder.

Metrics reported per model:
  - AIC, BIC        : Information criteria (lower = better)
  - RMSE, MAE, MAPE : Forecast accuracy on hold-out test set
  - R2              : Explained variance on test set  (max 1.0)
  - LB_p7, LB_p14   : Ljung-Box p-values (> 0.05 = residuals clean)

Usage:
  cd "d:\\Downloads\\Time Series\\TIME SERIES FINAL"
  python model_selection.py
"""

import os
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR   = r"d:\Downloads\Time Series\TIME SERIES FINAL\Cleaned"
OUTPUT_CSV = r"d:\Downloads\Time Series\TIME SERIES FINAL\model_selection_results.csv"

GAMES = {
    "EldenRing"    : "EldenRing_Cleaned.csv",
    "HuntShowdown" : "HuntShowdown_Cleaned.csv",
    "RDR2"         : "RDR2_Cleaned.csv",
    "ReadyorNot"   : "ReadyorNot_Cleaned.csv",
}

# Grid: p in {0,1,2,3}, d in {0,1}, q in {0,1,2}
P_RANGE = [0, 1, 2, 3]
D_RANGE = [0, 1]
Q_RANGE = [0, 1, 2]

TRAIN_RATIO = 0.80   # 80% train / 20% test (chronological)
MAX_ITER    = 300


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def prepare_data(csv_path):
    """Load and preprocess a game CSV file."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df = df.asfreq("D")
    df["Avg_Player"] = df["Avg_Player"].interpolate("linear")

    # Log-transform target
    df["Log_Player"] = np.log(df["Avg_Player"] + 1)

    # Scale exogenous variables
    df["Discount_Ratio"]      = df["Discount_%"] / 100.0
    df["Years_Since_Release"] = df["Days_Since_Release"] / 365.0

    fill_cols = ["Discount_Ratio", "Years_Since_Release", "Is_Weekend", "Is_Holiday"]
    df[fill_cols] = df[fill_cols].ffill().bfill()

    endog = df["Log_Player"]
    exog  = df[["Years_Since_Release", "Discount_Ratio", "Is_Weekend", "Is_Holiday"]]

    return df, endog, exog


def mape_score(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def fit_armax(endog_train, exog_train, endog_test, exog_test, p, d, q):
    """Fit ARMAX(p,d,q), forecast test set, compute all metrics."""
    try:
        model = ARIMA(
            endog=endog_train,
            exog=exog_train,
            order=(p, d, q),
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted = model.fit(
            method="statespace",
            method_kwargs={"maxiter": MAX_ITER},
        )

        # In-sample info criteria
        aic = fitted.aic
        bic = fitted.bic

        # Ljung-Box on residuals
        resid = fitted.resid.dropna()
        lb7   = acorr_ljungbox(resid, lags=[7],  return_df=True)["lb_pvalue"].values[0]
        lb14  = acorr_ljungbox(resid, lags=[14], return_df=True)["lb_pvalue"].values[0]

        # Out-of-sample forecast
        n_test   = len(endog_test)
        forecast = fitted.forecast(steps=n_test, exog=exog_test)

        rmse_val = np.sqrt(mean_squared_error(endog_test, forecast))
        mae_val  = mean_absolute_error(endog_test, forecast)
        mape_val = mape_score(endog_test.values, forecast.values)
        r2_val   = r2_score(endog_test, forecast)

        lb7_ok  = "OK"   if lb7  > 0.05 else "FAIL"
        lb14_ok = "OK"   if lb14 > 0.05 else "FAIL"

        return {
            "p": p, "d": d, "q": q,
            "AIC":     round(aic,      2),
            "BIC":     round(bic,      2),
            "RMSE":    round(rmse_val, 5),
            "MAE":     round(mae_val,  5),
            "MAPE":    round(mape_val, 3),
            "R2":      round(r2_val,   4),
            "LB_p7":   round(lb7,      4),
            "LB_p14":  round(lb14,     4),
            "LB7_OK":  lb7_ok,
            "LB14_OK": lb14_ok,
            "Converged": "OK",
            "_fitted":  fitted,
        }
    except Exception as e:
        return {
            "p": p, "d": d, "q": q,
            "AIC": None, "BIC": None,
            "RMSE": None, "MAE": None, "MAPE": None, "R2": None,
            "LB_p7": None, "LB_p14": None,
            "LB7_OK": "-", "LB14_OK": "-",
            "Converged": f"ERR:{str(e)[:35]}",
            "_fitted": None,
        }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    all_results = []
    grid = list(itertools.product(P_RANGE, D_RANGE, Q_RANGE))
    grid = [(p, d, q) for p, d, q in grid if not (p == 0 and d == 0 and q == 0)]

    print()
    print("=" * 80)
    print(f"  ARMAX MODEL SELECTION  --  {len(grid)} configs x {len(GAMES)} games = {len(grid)*len(GAMES)} fits")
    print("=" * 80)

    for game_name, csv_file in GAMES.items():
        csv_path = os.path.join(DATA_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found.")
            continue

        print()
        print("-" * 80)
        print(f"  GAME: {game_name}  ({csv_file})")
        print("-" * 80)

        df, endog, exog = prepare_data(csv_path)
        n_total = len(endog)
        n_train = int(n_total * TRAIN_RATIO)
        n_test  = n_total - n_train

        endog_train = endog.iloc[:n_train]
        endog_test  = endog.iloc[n_train:]
        exog_train  = exog.iloc[:n_train]
        exog_test   = exog.iloc[n_train:]

        print(f"  Total rows : {n_total}  |  Train: {n_train}  |  Test: {n_test}")
        print(f"  Date range : {df.index[0].date()} -> {df.index[-1].date()}")
        print(f"  Fitting {len(grid)} models...")
        print()

        game_results = []
        for i, (p, d, q) in enumerate(grid, 1):
            tag = f"ARMAX({p},{d},{q})"
            print(f"    [{i:02d}/{len(grid)}] {tag:<14} ...", end="", flush=True)
            res = fit_armax(endog_train, exog_train, endog_test, exog_test, p, d, q)
            res["Game"] = game_name
            game_results.append(res)
            if res["AIC"] is not None:
                print(f"  AIC={res['AIC']:>10.2f}  RMSE={res['RMSE']:.4f}  "
                      f"R2={res['R2']:.3f}  LB7={res['LB7_OK']:<5}  LB14={res['LB14_OK']}")
            else:
                print(f"  -> {res['Converged']}")

        # Sort by AIC
        valid = [r for r in game_results if r["AIC"] is not None]
        valid.sort(key=lambda r: r["AIC"])

        # Comparison table
        print()
        print(f"  {'='*74}")
        print(f"  TOP RESULTS for {game_name}  (sorted by AIC, lower = better)")
        print(f"  {'='*74}")
        print(f"  {'Rank':<5} {'Model':<15} {'AIC':>9} {'BIC':>9} {'RMSE':>8} {'MAE':>8} {'MAPE%':>7} {'R2':>7} {'LB7':>6} {'LB14':>6}")
        print(f"  {'-'*74}")
        for rank, r in enumerate(valid):
            prefix = f"  {'['+str(rank+1)+']':<5}"
            tag    = f"ARMAX({r['p']},{r['d']},{r['q']})"
            best_m = "*" if rank == 0 else (" " if rank >= 3 else " ")
            print(f"{prefix}{best_m}{tag:<14} {r['AIC']:>9.2f} {r['BIC']:>9.2f} "
                  f"{r['RMSE']:>8.4f} {r['MAE']:>8.4f} {r['MAPE']:>7.2f} "
                  f"{r['R2']:>7.3f} {r['LB7_OK']:>6} {r['LB14_OK']:>6}")

        # Full summary of best model
        if valid:
            best = valid[0]
            print()
            print(f"  ** BEST MODEL: ARMAX({best['p']},{best['d']},{best['q']})  "
                  f"AIC={best['AIC']}  R2={best['R2']}  LB7={best['LB7_OK']}  LB14={best['LB14_OK']}")
            print()
            print("  === FULL STATSMODELS SUMMARY (Best Model) ===")
            if best["_fitted"] is not None:
                print(best["_fitted"].summary().tables[1])

        all_results.extend(game_results)

    # Save CSV
    save_cols = ["Game", "p", "d", "q", "AIC", "BIC",
                 "RMSE", "MAE", "MAPE", "R2",
                 "LB_p7", "LB_p14", "LB7_OK", "LB14_OK", "Converged"]
    rows = [{k: r[k] for k in save_cols} for r in all_results]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print()
    print("=" * 80)
    print(f"  Results saved -> {OUTPUT_CSV}")
    print("=" * 80)

    # Global best-per-game table
    print()
    print("  GLOBAL BEST MODEL PER GAME (by AIC):")
    print()
    print(f"  {'Game':<16} {'Best Model':<16} {'AIC':>9} {'RMSE':>8} {'R2':>7} {'LB7':>6} {'LB14':>6}")
    print(f"  {'-'*72}")
    for game_name in GAMES:
        game_rows = [r for r in all_results if r["Game"] == game_name and r["AIC"] is not None]
        if game_rows:
            game_rows.sort(key=lambda r: r["AIC"])
            b   = game_rows[0]
            tag = f"ARMAX({b['p']},{b['d']},{b['q']})"
            print(f"  {game_name:<16} {tag:<16} {b['AIC']:>9.2f} "
                  f"{b['RMSE']:>8.4f} {b['R2']:>7.3f} {b['LB7_OK']:>6} {b['LB14_OK']:>6}")
    print()


if __name__ == "__main__":
    main()
