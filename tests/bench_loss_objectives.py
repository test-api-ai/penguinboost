"""Detailed accuracy benchmark for loss objectives vs LightGBM.

Benchmarks:
  - MSE  (baseline sanity check)
  - MAE  (L1 — was broken: leaf values ±1 due to constant hessian)
  - Huber (was broken: hessian=0 in linear region)
  - Quantile α=0.5 (median, was broken: wrong gradient sign)
  - Quantile α=0.1 / α=0.9 (tail quantiles)

Usage:
    python tests/bench_loss_objectives.py
"""

import time
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import penguinboost
from penguinboost import PenguinBoostRegressor, PenguinBoostQuantileRegressor

warnings.filterwarnings("ignore")

SEED = 42
N_SAMPLES = 10_000
N_FEATURES = 20
N_ESTIMATORS = 300
LEARNING_RATE = 0.05


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_data():
    X, y = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=10, noise=20.0, random_state=SEED,
    )
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def _pinball_loss(y_true, y_pred, alpha):
    r = y_true - y_pred
    return np.mean(np.where(r >= 0, alpha * r, (alpha - 1.0) * r))


def _section(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def _row(label, pb_val, lgb_val, lower_is_better=True):
    if lower_is_better:
        ratio = pb_val / lgb_val if lgb_val > 0 else float("inf")
        marker = "✓" if ratio <= 1.5 else "✗"
        vs = f"x{ratio:.2f} vs LGB"
    else:
        ratio = pb_val / lgb_val if lgb_val > 0 else float("inf")
        marker = "✓" if ratio >= 0.9 else "✗"
        vs = f"x{ratio:.2f} vs LGB"
    print(f"  {label:<28} PB={pb_val:>10.4f}  LGB={lgb_val:>10.4f}   {vs}  {marker}")


# ── MSE benchmark ─────────────────────────────────────────────────────────────

def bench_mse(X_tr, X_te, y_tr, y_te):
    _section("MSE (L2) — baseline")

    pb = PenguinBoostRegressor(
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED)
    t0 = time.perf_counter(); pb.fit(X_tr, y_tr); pb_fit = time.perf_counter() - t0
    pb_pred = pb.predict(X_te)

    lgb_m = lgb.LGBMRegressor(
        objective="mse", n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED, verbose=-1)
    t0 = time.perf_counter(); lgb_m.fit(X_tr, y_tr); lgb_fit = time.perf_counter() - t0
    lgb_pred = lgb_m.predict(X_te)

    pb_rmse  = np.sqrt(mean_squared_error(y_te, pb_pred))
    lgb_rmse = np.sqrt(mean_squared_error(y_te, lgb_pred))
    pb_mae   = mean_absolute_error(y_te, pb_pred)
    lgb_mae  = mean_absolute_error(y_te, lgb_pred)

    print(f"  Train time:  PB={pb_fit:.2f}s  LGB={lgb_fit:.2f}s")
    _row("RMSE  (lower better)", pb_rmse, lgb_rmse)
    _row("MAE   (lower better)", pb_mae,  lgb_mae)

    return dict(objective="mse", pb_rmse=pb_rmse, lgb_rmse=lgb_rmse,
                pb_mae=pb_mae, lgb_mae=lgb_mae)


# ── MAE benchmark ─────────────────────────────────────────────────────────────

def bench_mae(X_tr, X_te, y_tr, y_te):
    _section("MAE (L1) — was broken (leaf values saturated at ±1)")

    pb = PenguinBoostRegressor(
        objective="mae",
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED)
    t0 = time.perf_counter(); pb.fit(X_tr, y_tr); pb_fit = time.perf_counter() - t0
    pb_pred = pb.predict(X_te)

    lgb_m = lgb.LGBMRegressor(
        objective="mae", n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED, verbose=-1)
    t0 = time.perf_counter(); lgb_m.fit(X_tr, y_tr); lgb_fit = time.perf_counter() - t0
    lgb_pred = lgb_m.predict(X_te)

    pb_rmse  = np.sqrt(mean_squared_error(y_te, pb_pred))
    lgb_rmse = np.sqrt(mean_squared_error(y_te, lgb_pred))
    pb_mae   = mean_absolute_error(y_te, pb_pred)
    lgb_mae  = mean_absolute_error(y_te, lgb_pred)

    print(f"  Train time:  PB={pb_fit:.2f}s  LGB={lgb_fit:.2f}s")
    _row("RMSE  (lower better)", pb_rmse, lgb_rmse)
    _row("MAE   (lower better)", pb_mae,  lgb_mae)
    print(f"  Pred range: PB=[{pb_pred.min():.1f}, {pb_pred.max():.1f}]  "
          f"LGB=[{lgb_pred.min():.1f}, {lgb_pred.max():.1f}]  "
          f"true=[{y_te.min():.1f}, {y_te.max():.1f}]")

    return dict(objective="mae", pb_rmse=pb_rmse, lgb_rmse=lgb_rmse,
                pb_mae=pb_mae, lgb_mae=lgb_mae)


# ── Huber benchmark ───────────────────────────────────────────────────────────

def bench_huber(X_tr, X_te, y_tr, y_te):
    _section("Huber (delta=1.0) — was broken (hessian=0 in linear region)")

    pb = PenguinBoostRegressor(
        objective="huber", huber_delta=1.0,
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED)
    t0 = time.perf_counter(); pb.fit(X_tr, y_tr); pb_fit = time.perf_counter() - t0
    pb_pred = pb.predict(X_te)

    # LightGBM Huber: alpha sets the quantile of |residuals| used to compute
    # the adaptive delta threshold (default 0.9). We use the default here for
    # a fair comparison since LGB's alpha != our fixed delta parameter.
    lgb_m = lgb.LGBMRegressor(
        objective="huber",  # alpha=0.9 default: delta = 90th percentile of |residuals|
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED, verbose=-1)
    t0 = time.perf_counter(); lgb_m.fit(X_tr, y_tr); lgb_fit = time.perf_counter() - t0
    lgb_pred = lgb_m.predict(X_te)

    pb_rmse  = np.sqrt(mean_squared_error(y_te, pb_pred))
    lgb_rmse = np.sqrt(mean_squared_error(y_te, lgb_pred))
    pb_mae   = mean_absolute_error(y_te, pb_pred)
    lgb_mae  = mean_absolute_error(y_te, lgb_pred)

    print(f"  Train time:  PB={pb_fit:.2f}s  LGB={lgb_fit:.2f}s")
    _row("RMSE  (lower better)", pb_rmse, lgb_rmse)
    _row("MAE   (lower better)", pb_mae,  lgb_mae)
    print(f"  Pred range: PB=[{pb_pred.min():.1f}, {pb_pred.max():.1f}]  "
          f"LGB=[{lgb_pred.min():.1f}, {lgb_pred.max():.1f}]")

    return dict(objective="huber", pb_rmse=pb_rmse, lgb_rmse=lgb_rmse,
                pb_mae=pb_mae, lgb_mae=lgb_mae)


# ── Quantile benchmarks ────────────────────────────────────────────────────────

def bench_quantile(X_tr, X_te, y_tr, y_te, alpha):
    _section(f"Quantile α={alpha} — was broken (wrong gradient sign)")

    pb = PenguinBoostQuantileRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED)
    t0 = time.perf_counter(); pb.fit(X_tr, y_tr); pb_fit = time.perf_counter() - t0
    pb_pred = pb.predict(X_te)

    lgb_m = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
        max_depth=6, reg_lambda=1.0, random_state=SEED, verbose=-1)
    t0 = time.perf_counter(); lgb_m.fit(X_tr, y_tr); lgb_fit = time.perf_counter() - t0
    lgb_pred = lgb_m.predict(X_te)

    pb_loss  = _pinball_loss(y_te, pb_pred,  alpha)
    lgb_loss = _pinball_loss(y_te, lgb_pred, alpha)
    pb_cov   = np.mean(pb_pred  >= y_te)
    lgb_cov  = np.mean(lgb_pred >= y_te)

    print(f"  Train time:  PB={pb_fit:.2f}s  LGB={lgb_fit:.2f}s")
    _row(f"Pinball loss α={alpha} (lower)", pb_loss, lgb_loss)
    print(f"  Coverage (pred≥y): PB={pb_cov:.3f}  LGB={lgb_cov:.3f}  "
          f"target={1-alpha:.3f}")
    print(f"  Pred range: PB=[{pb_pred.min():.1f}, {pb_pred.max():.1f}]  "
          f"LGB=[{lgb_pred.min():.1f}, {lgb_pred.max():.1f}]")

    return dict(objective=f"quantile_a{alpha}",
                pb_loss=pb_loss, lgb_loss=lgb_loss,
                pb_coverage=pb_cov, lgb_coverage=lgb_cov)


# ── Summary ────────────────────────────────────────────────────────────────────

def _summary(results):
    _section("SUMMARY  (PB / LGB ratio — 1.0 means identical)")
    print(f"\n  {'Objective':<28} {'Metric':<18} {'PB':>10} {'LGB':>10} {'Ratio':>8}")
    print("  " + "-" * 76)
    for r in results:
        obj = r["objective"]
        if "pb_rmse" in r:
            ratio = r["pb_rmse"] / r["lgb_rmse"]
            status = "✓" if ratio < 2.0 else "✗ WORSE"
            print(f"  {obj:<28} {'RMSE':<18} {r['pb_rmse']:>10.4f} {r['lgb_rmse']:>10.4f} "
                  f"{ratio:>8.2f}x  {status}")
        if "pb_mae" in r and obj in ("mae", "huber"):
            ratio = r["pb_mae"] / r["lgb_mae"]
            status = "✓" if ratio < 2.0 else "✗ WORSE"
            print(f"  {obj:<28} {'MAE':<18} {r['pb_mae']:>10.4f} {r['lgb_mae']:>10.4f} "
                  f"{ratio:>8.2f}x  {status}")
        if "pb_loss" in r:
            ratio = r["pb_loss"] / r["lgb_loss"]
            status = "✓" if ratio < 2.0 else "✗ WORSE"
            print(f"  {obj:<28} {'Pinball loss':<18} {r['pb_loss']:>10.4f} {r['lgb_loss']:>10.4f} "
                  f"{ratio:>8.2f}x  {status}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("PenguinBoost Loss Objectives — Detailed Accuracy Benchmark")
    print(f"penguinboost {penguinboost.__version__}  vs  lightgbm {lgb.__version__}")
    print(f"n_samples={N_SAMPLES}, n_features={N_FEATURES}, "
          f"n_estimators={N_ESTIMATORS}, lr={LEARNING_RATE}")

    X_tr, X_te, y_tr, y_te = _make_data()
    print(f"y range: [{y_te.min():.1f}, {y_te.max():.1f}]  "
          f"std={y_te.std():.1f}")

    results = []
    results.append(bench_mse(X_tr, X_te, y_tr, y_te))
    results.append(bench_mae(X_tr, X_te, y_tr, y_te))
    results.append(bench_huber(X_tr, X_te, y_tr, y_te))
    results.append(bench_quantile(X_tr, X_te, y_tr, y_te, alpha=0.5))
    results.append(bench_quantile(X_tr, X_te, y_tr, y_te, alpha=0.1))
    results.append(bench_quantile(X_tr, X_te, y_tr, y_te, alpha=0.9))

    _summary(results)


if __name__ == "__main__":
    main()
