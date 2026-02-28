"""Performance benchmark: PenguinBoost vs LightGBM.

Compares training time, prediction time, and predictive accuracy
across regression and classification tasks at multiple dataset sizes.

Usage:
    python tests/bench_vs_lightgbm.py
"""

import time
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
)

from penguinboost import PenguinBoostRegressor, PenguinBoostClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ── Shared hyperparameters (equivalent settings) ─────────────────────────────
COMMON = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
)

PB_REG_PARAMS = dict(**COMMON, reg_lambda=1.0, max_bins=255)
PB_CLF_PARAMS = dict(**COMMON, reg_lambda=1.0, max_bins=255)

LGB_REG_PARAMS = dict(
    n_estimators=COMMON["n_estimators"],
    learning_rate=COMMON["learning_rate"],
    max_depth=COMMON["max_depth"],
    random_state=RANDOM_STATE,
    reg_lambda=1.0,
    max_bin=255,
    verbose=-1,
)
LGB_CLF_PARAMS = dict(**LGB_REG_PARAMS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fit_time(model, X_tr, y_tr):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    return time.perf_counter() - t0


def _predict_time(model, X_te):
    t0 = time.perf_counter()
    preds = model.predict(X_te)
    return time.perf_counter() - t0, preds


def _bar(label, value, best, width=30):
    ratio = value / best if best > 0 else 1.0
    filled = int(min(ratio, 3.0) * width / 3)
    bar = "#" * filled + "-" * (width - filled)
    marker = " <-- best" if abs(ratio - 1.0) < 1e-9 else f"  (x{ratio:.2f})"
    return f"  {label:<14} [{bar}] {value:>9.4f}{marker}"


def _section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Regression benchmark ──────────────────────────────────────────────────────

def bench_regression(n_samples, n_features):
    _section(f"REGRESSION  n_samples={n_samples:,}  n_features={n_features}")

    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(n_features // 2, 5),
        noise=10.0, random_state=RANDOM_STATE,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # PenguinBoost
    pb = PenguinBoostRegressor(**PB_REG_PARAMS)
    pb_fit = _fit_time(pb, X_tr, y_tr)
    pb_pred_time, pb_preds = _predict_time(pb, X_te)
    pb_rmse = np.sqrt(mean_squared_error(y_te, pb_preds))
    pb_r2 = r2_score(y_te, pb_preds)

    # LightGBM
    lgbm = lgb.LGBMRegressor(**LGB_REG_PARAMS)
    lgbm_fit = _fit_time(lgbm, X_tr, y_tr)
    lgbm_pred_time, lgbm_preds = _predict_time(lgbm, X_te)
    lgbm_rmse = np.sqrt(mean_squared_error(y_te, lgbm_preds))
    lgbm_r2 = r2_score(y_te, lgbm_preds)

    # Display
    best_fit  = min(pb_fit, lgbm_fit)
    best_pred = min(pb_pred_time, lgbm_pred_time)
    best_rmse = min(pb_rmse, lgbm_rmse)

    print("\n[Training time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_fit,   best_fit))
    print(_bar("LightGBM",    lgbm_fit,  best_fit))

    print("\n[Prediction time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_pred_time,   best_pred))
    print(_bar("LightGBM",    lgbm_pred_time,  best_pred))

    print("\n[RMSE — lower is better]")
    print(_bar("PenguinBoost", pb_rmse,   best_rmse))
    print(_bar("LightGBM",    lgbm_rmse,  best_rmse))

    print("\n[R² score — higher is better]")
    best_r2 = max(pb_r2, lgbm_r2)
    # invert so that _bar still means "closer to 1 is best"
    print(f"  PenguinBoost  {pb_r2:.4f}")
    print(f"  LightGBM      {lgbm_r2:.4f}")

    return {
        "task": "regression",
        "n_samples": n_samples,
        "n_features": n_features,
        "pb_fit_sec":  pb_fit,
        "lgb_fit_sec": lgbm_fit,
        "pb_pred_sec":  pb_pred_time,
        "lgb_pred_sec": lgbm_pred_time,
        "pb_rmse":  pb_rmse,
        "lgb_rmse": lgbm_rmse,
        "pb_r2":  pb_r2,
        "lgb_r2": lgbm_r2,
    }


# ── Binary classification benchmark ──────────────────────────────────────────

def bench_binary_classification(n_samples, n_features):
    _section(f"BINARY CLASSIFICATION  n_samples={n_samples:,}  n_features={n_features}")

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(n_features // 2, 5),
        n_redundant=2, random_state=RANDOM_STATE,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # PenguinBoost
    pb = PenguinBoostClassifier(**PB_CLF_PARAMS)
    pb_fit = _fit_time(pb, X_tr, y_tr)
    pb_pred_time, pb_labels = _predict_time(pb, X_te)
    pb_proba = pb.predict_proba(X_te)[:, 1]
    pb_acc = accuracy_score(y_te, pb_labels)
    pb_auc = roc_auc_score(y_te, pb_proba)

    # LightGBM
    lgbm = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
    lgbm_fit = _fit_time(lgbm, X_tr, y_tr)
    lgbm_pred_time, lgbm_labels = _predict_time(lgbm, X_te)
    lgbm_proba = lgbm.predict_proba(X_te)[:, 1]
    lgbm_acc = accuracy_score(y_te, lgbm_labels)
    lgbm_auc = roc_auc_score(y_te, lgbm_proba)

    # Display
    best_fit  = min(pb_fit, lgbm_fit)
    best_pred = min(pb_pred_time, lgbm_pred_time)

    print("\n[Training time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_fit,   best_fit))
    print(_bar("LightGBM",    lgbm_fit,  best_fit))

    print("\n[Prediction time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_pred_time,   best_pred))
    print(_bar("LightGBM",    lgbm_pred_time,  best_pred))

    print("\n[Accuracy — higher is better]")
    print(f"  PenguinBoost  {pb_acc:.4f}")
    print(f"  LightGBM      {lgbm_acc:.4f}")

    print("\n[ROC-AUC — higher is better]")
    print(f"  PenguinBoost  {pb_auc:.4f}")
    print(f"  LightGBM      {lgbm_auc:.4f}")

    return {
        "task": "binary_clf",
        "n_samples": n_samples,
        "n_features": n_features,
        "pb_fit_sec":  pb_fit,
        "lgb_fit_sec": lgbm_fit,
        "pb_pred_sec":  pb_pred_time,
        "lgb_pred_sec": lgbm_pred_time,
        "pb_acc":  pb_acc,
        "lgb_acc": lgbm_acc,
        "pb_auc":  pb_auc,
        "lgb_auc": lgbm_auc,
    }


# ── Multiclass classification benchmark ──────────────────────────────────────

def bench_multiclass_classification(n_samples, n_features, n_classes=5):
    _section(
        f"MULTICLASS CLASSIFICATION  n_samples={n_samples:,}  "
        f"n_features={n_features}  n_classes={n_classes}"
    )

    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(n_features // 2, n_classes + 2),
        n_redundant=2, n_classes=n_classes,
        n_clusters_per_class=1, random_state=RANDOM_STATE,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # PenguinBoost
    pb = PenguinBoostClassifier(**PB_CLF_PARAMS)
    pb_fit = _fit_time(pb, X_tr, y_tr)
    pb_pred_time, pb_labels = _predict_time(pb, X_te)
    pb_acc = accuracy_score(y_te, pb_labels)

    # LightGBM
    lgbm = lgb.LGBMClassifier(**LGB_CLF_PARAMS)
    lgbm_fit = _fit_time(lgbm, X_tr, y_tr)
    lgbm_pred_time, lgbm_labels = _predict_time(lgbm, X_te)
    lgbm_acc = accuracy_score(y_te, lgbm_labels)

    best_fit  = min(pb_fit, lgbm_fit)
    best_pred = min(pb_pred_time, lgbm_pred_time)

    print("\n[Training time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_fit,   best_fit))
    print(_bar("LightGBM",    lgbm_fit,  best_fit))

    print("\n[Prediction time (sec) — lower is better]")
    print(_bar("PenguinBoost", pb_pred_time,   best_pred))
    print(_bar("LightGBM",    lgbm_pred_time,  best_pred))

    print("\n[Accuracy — higher is better]")
    print(f"  PenguinBoost  {pb_acc:.4f}")
    print(f"  LightGBM      {lgbm_acc:.4f}")

    return {
        "task": "multiclass_clf",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "pb_fit_sec":  pb_fit,
        "lgb_fit_sec": lgbm_fit,
        "pb_pred_sec":  pb_pred_time,
        "lgb_pred_sec": lgbm_pred_time,
        "pb_acc":  pb_acc,
        "lgb_acc": lgbm_acc,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def _summary(results):
    _section("SUMMARY")

    W = [10, 12, 10, 10, 10, 10, 10, 10]
    header = (
        f"{'Task':<10} {'Dataset':<12} "
        f"{'PB_fit':>10} {'LGB_fit':>10} "
        f"{'PB_pred':>10} {'LGB_pred':>10} "
        f"{'PB_metric':>10} {'LGB_metric':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ds = f"{r['n_samples']}x{r['n_features']}"
        if r["task"] == "regression":
            m_label = "RMSE"
            pb_m  = r["pb_rmse"]
            lgb_m = r["lgb_rmse"]
        else:
            m_label = "Acc"
            pb_m  = r["pb_acc"]
            lgb_m = r["lgb_acc"]

        task_short = {"regression": "Regr", "binary_clf": "Bin", "multiclass_clf": "Multi"}[r["task"]]
        print(
            f"{task_short:<10} {ds:<12} "
            f"{r['pb_fit_sec']:>10.3f} {r['lgb_fit_sec']:>10.3f} "
            f"{r['pb_pred_sec']:>10.4f} {r['lgb_pred_sec']:>10.4f} "
            f"{pb_m:>10.4f} {lgb_m:>10.4f}"
        )

    print()
    # Speed-up ratios
    fit_speedup  = np.mean([r["lgb_fit_sec"]  / r["pb_fit_sec"]  for r in results])
    pred_speedup = np.mean([r["lgb_pred_sec"] / r["pb_pred_sec"] for r in results])
    print(f"Avg training-time ratio  (LGB / PB): {fit_speedup:.2f}x")
    print(f"Avg prediction-time ratio (LGB / PB): {pred_speedup:.2f}x")
    print("(>1.0 means PenguinBoost is faster; <1.0 means LightGBM is faster)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("PenguinBoost vs LightGBM — Performance Benchmark")
    print(f"LightGBM version : {lgb.__version__}")
    import penguinboost
    print(f"PenguinBoost version: {penguinboost.__version__}")

    results = []

    # --- Regression ---
    for n_samples, n_features in [(5_000, 20), (20_000, 50), (50_000, 100)]:
        results.append(bench_regression(n_samples, n_features))

    # --- Binary classification ---
    for n_samples, n_features in [(5_000, 20), (20_000, 50)]:
        results.append(bench_binary_classification(n_samples, n_features))

    # --- Multiclass classification ---
    results.append(bench_multiclass_classification(10_000, 30, n_classes=5))

    _summary(results)


if __name__ == "__main__":
    main()
