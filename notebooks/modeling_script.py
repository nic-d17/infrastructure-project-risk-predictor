"""
Modeling pipeline for PPI distress classifier.
Steps:
  1. Load features_train.parquet / features_test.parquet
  2. Baseline models: constant-rate, logistic regression, random forest
  3. Primary models: XGBoost and LightGBM with Optuna tuning (100 trials each)
  4. Calibration: Platt scaling on the best model
  5. Reporting: model comparison table, calibration curve, per-fold AUC

Evaluation: 5-fold stratified CV on train (primary), out-of-time test (sanity check only).
See docs/PROJECT_PLAN.md for full rationale on evaluation strategy.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT     = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
MDL_DIR  = ROOT / "models"
FIG_DIR  = ROOT / "notebooks" / "figures"
MDL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CV_FOLDS   = 5
OPTUNA_TRIALS = 100
RANDOM_STATE  = 42

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              list[str]]:
    train = pd.read_parquet(PROC_DIR / "features_train.parquet")
    test  = pd.read_parquet(PROC_DIR / "features_test.parquet")

    feature_cols = [c for c in train.columns if c not in ("FCY", "target")]

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train["target"].values.astype(np.int32)
    X_test  = test[feature_cols].values.astype(np.float32)
    y_test  = test["target"].values.astype(np.int32)

    print(f"Train: {X_train.shape}  distress={y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"Test:  {X_test.shape}   distress={y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"Features: {len(feature_cols)}")
    return X_train, y_train, X_test, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CROSS-VALIDATION HELPER
# ═══════════════════════════════════════════════════════════════════════════

def cv_metrics(model, X, y, n_folds=CV_FOLDS) -> dict:
    """Run stratified k-fold CV, return mean+std of AUC, Brier, PR-AUC."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    aucs, briers, pr_aucs, fold_aucs = [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]

        auc    = roc_auc_score(y_va, y_prob)
        brier  = brier_score_loss(y_va, y_prob)
        pr_auc = average_precision_score(y_va, y_prob)

        aucs.append(auc)
        briers.append(brier)
        pr_aucs.append(pr_auc)
        fold_aucs.append((fold + 1, auc))

    return {
        "auc_mean":    np.mean(aucs),
        "auc_std":     np.std(aucs),
        "brier_mean":  np.mean(briers),
        "brier_std":   np.std(briers),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std":  np.std(pr_aucs),
        "fold_aucs":   fold_aucs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — BASELINE MODELS
# ═══════════════════════════════════════════════════════════════════════════

def run_baselines(X_train, y_train) -> dict:
    results = {}

    # Constant-rate baseline: always predict the training base rate
    base_rate = y_train.mean()
    n = len(y_train)
    # Brier for constant prediction: var(y)
    const_brier = brier_score_loss(y_train, np.full(n, base_rate))
    # AUC for constant prediction = 0.5 by definition
    # PR-AUC for constant = base rate (all predictions equal → random ranking)
    results["Constant baseline"] = {
        "auc_mean":    0.5000,   "auc_std":     0.0,
        "brier_mean":  const_brier, "brier_std":  0.0,
        "pr_auc_mean": base_rate,   "pr_auc_std": 0.0,
        "fold_aucs":   [],
    }
    print(f"  Constant baseline: AUC=0.500  Brier={const_brier:.4f}  PR-AUC={base_rate:.4f}")

    # Logistic regression with L2 regularisation (scaled features)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ])
    lr_res = cv_metrics(lr, X_train, y_train)
    results["Logistic Regression"] = lr_res
    print(f"  LogReg: AUC={lr_res['auc_mean']:.4f}±{lr_res['auc_std']:.4f}  "
          f"Brier={lr_res['brier_mean']:.4f}±{lr_res['brier_std']:.4f}  "
          f"PR-AUC={lr_res['pr_auc_mean']:.4f}±{lr_res['pr_auc_std']:.4f}")

    # Random forest with default hyperparameters
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    rf_res = cv_metrics(rf, X_train, y_train)
    results["Random Forest"] = rf_res
    print(f"  RF:     AUC={rf_res['auc_mean']:.4f}±{rf_res['auc_std']:.4f}  "
          f"Brier={rf_res['brier_mean']:.4f}±{rf_res['brier_std']:.4f}  "
          f"PR-AUC={rf_res['pr_auc_mean']:.4f}±{rf_res['pr_auc_std']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════

def _xgb_objective(trial, X, y):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":  (y == 0).sum() / (y == 1).sum(),
        "use_label_encoder": False,
        "eval_metric":       "auc",
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "tree_method":       "hist",
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, va in cv.split(X, y):
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]))
    return np.mean(aucs)


def _lgb_objective(trial, X, y):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":  (y == 0).sum() / (y == 1).sum(),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, va in cv.split(X, y):
        model.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], model.predict_proba(X[va])[:, 1]))
    return np.mean(aucs)


def tune_xgboost(X, y, n_trials=OPTUNA_TRIALS) -> dict:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _xgb_objective(t, X, y),
                   n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    })
    print(f"  XGBoost best trial AUC: {study.best_value:.4f}")
    return best


def tune_lightgbm(X, y, n_trials=OPTUNA_TRIALS) -> dict:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _lgb_objective(t, X, y),
                   n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({
        "scale_pos_weight": (y == 0).sum() / (y == 1).sum(),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    })
    print(f"  LightGBM best trial AUC: {study.best_value:.4f}")
    return best


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRIMARY MODELS
# ═══════════════════════════════════════════════════════════════════════════

def run_primary_models(X_train, y_train) -> tuple[dict, dict, dict]:
    print("  Tuning XGBoost (100 trials)...")
    xgb_params = tune_xgboost(X_train, y_train)
    xgb_model  = xgb.XGBClassifier(**xgb_params)
    xgb_res    = cv_metrics(xgb_model, X_train, y_train)
    print(f"  XGBoost CV: AUC={xgb_res['auc_mean']:.4f}±{xgb_res['auc_std']:.4f}  "
          f"Brier={xgb_res['brier_mean']:.4f}±{xgb_res['brier_std']:.4f}  "
          f"PR-AUC={xgb_res['pr_auc_mean']:.4f}±{xgb_res['pr_auc_std']:.4f}")

    print("  Tuning LightGBM (100 trials)...")
    lgb_params = tune_lightgbm(X_train, y_train)
    lgb_model  = lgb.LGBMClassifier(**lgb_params)
    lgb_res    = cv_metrics(lgb_model, X_train, y_train)
    print(f"  LightGBM CV: AUC={lgb_res['auc_mean']:.4f}±{lgb_res['auc_std']:.4f}  "
          f"Brier={lgb_res['brier_mean']:.4f}±{lgb_res['brier_std']:.4f}  "
          f"PR-AUC={lgb_res['pr_auc_mean']:.4f}±{lgb_res['pr_auc_std']:.4f}")

    return xgb_res, lgb_res, {"xgb": xgb_params, "lgb": lgb_params}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_best_model(best_model_cls, best_params, X_train, y_train,
                         X_test, y_test) -> tuple:
    """
    Fit the best model on full train, apply Platt scaling, produce calibration curve.
    Returns: (uncal_model, cal_model, fig)
    """
    # Fit uncalibrated on full train
    uncal_model = best_model_cls(**best_params)
    uncal_model.fit(X_train, y_train)

    # Calibrate with Platt scaling (sigmoid) using 5-fold CV
    cal_model = CalibratedClassifierCV(best_model_cls(**best_params),
                                       method="sigmoid", cv=5)
    cal_model.fit(X_train, y_train)

    # Produce calibration curves on train (enough data for reliable bins)
    prob_uncal = uncal_model.predict_proba(X_train)[:, 1]
    prob_cal   = cal_model.predict_proba(X_train)[:, 1]

    frac_pos_u, mean_pred_u = calibration_curve(y_train, prob_uncal, n_bins=10)
    frac_pos_c, mean_pred_c = calibration_curve(y_train, prob_cal,   n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred_u, frac_pos_u, "o-", color="#d73027", label="Uncalibrated")
    ax.plot(mean_pred_c, frac_pos_c, "s-", color="#4575b4", label="Platt-scaled")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (train set, 10 bins)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_calibration_curve.png", dpi=150)
    plt.show()
    print("Saved: notebooks/figures/03_calibration_curve.png")

    # Calibration metrics
    brier_uncal = brier_score_loss(y_train, prob_uncal)
    brier_cal   = brier_score_loss(y_train, prob_cal)
    print(f"  Brier (uncalibrated): {brier_uncal:.4f}")
    print(f"  Brier (calibrated):   {brier_cal:.4f}")

    return uncal_model, cal_model, fig


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — FOLD STABILITY CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_fold_stability(results: dict) -> None:
    """Bar chart of per-fold AUC for the best model."""
    # Find winner (highest mean AUC among primary models)
    primary_names = ["XGBoost", "LightGBM"]
    best_name = max(
        [n for n in primary_names if n in results],
        key=lambda n: results[n]["auc_mean"]
    )
    fold_aucs = results[best_name]["fold_aucs"]
    folds, aucs = zip(*fold_aucs)
    overall_mean = results[best_name]["auc_mean"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(folds, aucs, color="#4575b4", edgecolor="white", width=0.6)
    ax.axhline(overall_mean, color="#d73027", lw=1.5, linestyle="--",
               label=f"Mean = {overall_mean:.4f}")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, auc + 0.003,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{best_name} — per-fold AUC (5-fold stratified CV)")
    ax.set_ylim(max(0, min(aucs) - 0.05), min(1, max(aucs) + 0.05))
    ax.set_xticks(list(folds))
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_fold_stability.png", dpi=150)
    plt.show()
    print(f"Saved: notebooks/figures/03_fold_stability.png")
    return best_name


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: dict) -> None:
    print("\n" + "═" * 90)
    print("  MODEL COMPARISON TABLE (5-fold stratified CV, train set)")
    print("═" * 90)
    print(f"  {'Model':30s}  {'AUC':>14s}  {'Brier':>14s}  {'PR-AUC':>14s}")
    print("  " + "-" * 86)
    order = ["Constant baseline", "Logistic Regression", "Random Forest",
             "XGBoost", "LightGBM"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        if r["auc_std"] == 0:
            auc_s    = f"{r['auc_mean']:.4f}"
            brier_s  = f"{r['brier_mean']:.4f}"
            prauc_s  = f"{r['pr_auc_mean']:.4f}"
        else:
            auc_s    = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
            brier_s  = f"{r['brier_mean']:.4f} ± {r['brier_std']:.4f}"
            prauc_s  = f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}"
        print(f"  {name:30s}  {auc_s:>14s}  {brier_s:>14s}  {prauc_s:>14s}")
    print("═" * 90)

    # AUC lift over random baseline
    best_primary = max(["XGBoost", "LightGBM"],
                       key=lambda n: results.get(n, {}).get("auc_mean", 0))
    best_auc = results[best_primary]["auc_mean"]
    print(f"\n  Best model: {best_primary}")
    print(f"  AUC lift over random: {best_auc - 0.5:+.4f}")
    print(f"  AUC lift over logistic: "
          f"{best_auc - results['Logistic Regression']['auc_mean']:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("  MODELING PIPELINE — PPI Distress Classifier")
    print("═" * 70)

    print("\n─── Step 1: Load features ────────────────────────────────────")
    X_train, y_train, X_test, y_test, feature_cols = load_features()

    print("\n─── Step 2: Baseline models ──────────────────────────────────")
    results = run_baselines(X_train, y_train)

    print("\n─── Step 3: Primary models (Optuna tuning) ───────────────────")
    xgb_res, lgb_res, best_params = run_primary_models(X_train, y_train)
    results["XGBoost"]   = xgb_res
    results["LightGBM"]  = lgb_res

    print("\n─── Step 4: Comparison table ─────────────────────────────────")
    print_comparison_table(results)

    print("\n─── Step 5: Fold stability ───────────────────────────────────")
    best_name = plot_fold_stability(results)

    print("\n─── Step 6: Calibration ──────────────────────────────────────")
    if best_name == "XGBoost":
        best_cls = xgb.XGBClassifier
    else:
        best_cls = lgb.LGBMClassifier
    uncal, cal, _ = calibrate_best_model(
        best_cls, best_params[best_name.lower()[:3]],
        X_train, y_train, X_test, y_test
    )

    print("\n─── Step 7: Save artifacts ───────────────────────────────────")
    with open(MDL_DIR / "best_model_uncalibrated.pkl", "wb") as f:
        pickle.dump(uncal, f)
    with open(MDL_DIR / "best_model_calibrated.pkl", "wb") as f:
        pickle.dump(cal, f)
    with open(MDL_DIR / "best_params.json", "w") as f:
        # Serialize params (scale_pos_weight is float, safe for JSON)
        p = {k: (float(v) if isinstance(v, (np.floating, float)) else
                 int(v)   if isinstance(v, (np.integer, int)) else v)
             for k, v in best_params[best_name.lower()[:3]].items()}
        json.dump({"model": best_name, "params": p}, f, indent=2)
    with open(MDL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"  Saved: models/best_model_uncalibrated.pkl")
    print(f"  Saved: models/best_model_calibrated.pkl")
    print(f"  Saved: models/best_params.json")
    print(f"  Saved: models/feature_cols.json")
    print("\nDone. Show user: comparison table, calibration curve, fold stability.")
