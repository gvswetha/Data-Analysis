#!/usr/bin/env python3
"""
interpretable_gbm_shap_lime.py

Full, clean pipeline:
- Load CSV
- Preprocess (impute, scale numeric; impute + one-hot encode categorical)
- Optional SMOTE (if imblearn installed)
- Train a GBM classifier (XGBoost by default)
- Evaluate (AUC, Precision, Recall, classification report)
- SHAP global (summary + dependence plots for top K features)
- LIME local explanations for 3 test cases (save images)
- Save model, preprocessor, metrics, and a short report

Notes:
- Don't run pip install inside this script.
- If you don't have imblearn or lime or shap installed, pip install them outside:
  pip install xgboost shap lime imbalanced-learn matplotlib scikit-learn joblib
"""

import os
import argparse
import json
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
    auc as calc_auc
)

import joblib

# Model libraries
import xgboost as xgb

# Explainability libraries
import shap
from lime.lime_tabular import LimeTabularExplainer

# Try to import imblearn (SMOTE) — optional
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# -------------------------
# Utility functions
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Interpretable GBM with SHAP and LIME")
    p.add_argument("--data", required=True, help="Path to input CSV file")
    p.add_argument("--target_col", default="target", help="Name of binary target column (0/1)")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--use_smote", action="store_true", help="Apply SMOTE to training set (requires imblearn)")
    p.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost", help="Which GBM to use")
    p.add_argument("--shap_top_k", type=int, default=5, help="Number of top SHAP features to show dependence plots for")
    return p.parse_args()

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def detect_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    features = [c for c in df.columns if c != target_col]
    categorical = [c for c in features if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
    numeric = [c for c in features if c not in categorical]
    return numeric, categorical

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    # Numeric pipeline: impute median, scale
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    # Categorical pipeline: impute constant then one-hot encode (return dense arrays)
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    return preprocessor

def build_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    # After fitting the preprocessor, extract names in the transformed order
    names = []
    # numeric names are same as original
    names.extend(numeric_features)

    # categorical expanded names via OneHotEncoder
    # locate the 'cat' transformer
    try:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        ohe = cat_pipeline.named_steps["ohe"]
        cat_cols = categorical_features
        ohe_names = ohe.get_feature_names_out(cat_cols)
        names.extend(list(ohe_names))
    except Exception:
        # fallback: append categorical column names themselves
        names.extend(categorical_features)
    return names

# -------------------------
# Training / evaluation
# -------------------------
def train_gbm(X_train, y_train, X_val, y_val, args):
    if args.model == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=args.random_state,
            n_jobs=-1
        )
        # Try calling fit with early stopping; some xgboost versions / wrappers
        # don't accept the `early_stopping_rounds` kwarg. Fall back gracefully.
        try:
            clf.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            # older/newer wrapper may reject early_stopping_rounds; try with eval_set only
            try:
                print("XGBClassifier.fit: 'early_stopping_rounds' not supported; trying without it.")
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            except TypeError:
                # As a last resort, call fit with only X and y
                print("XGBClassifier.fit: 'eval_set' also not supported; calling fit(X, y) without extra args.")
                clf.fit(X_train, y_train)
        return clf
    else:
        # LightGBM fallback if user chooses it (requires lightgbm installed)
        try:
            import lightgbm as lgb
        except Exception:
            raise RuntimeError("lightgbm not installed; install via `pip install lightgbm` or use --model xgboost")
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=args.random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        return clf

def evaluate_model(clf, X_test, y_test) -> dict:
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    report = classification_report(y_test, pred, digits=4)
    print("=== Classification Report ===")
    print(report)
    return {"auc": float(auc), "precision": float(precision), "recall": float(recall), "report": report, "y_proba": proba.tolist()}

# -------------------------
# SHAP analysis
# -------------------------
def shap_global_analysis(clf, preprocessor, X_background_raw: pd.DataFrame, feature_names: List[str], output_dir: str, top_k: int = 5):
    """Compute SHAP values and save summary + dependence plots for top_k features."""
    # Transform background data (must use the preprocessor fitted previously)
    X_bg_trans = preprocessor.transform(X_background_raw)
    # Use TreeExplainer for tree models
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception:
        explainer = shap.Explainer(clf)  # fallback

    shap_values = explainer.shap_values(X_bg_trans)
    # shap_values may be array or list depending on explainer; convert to 2D (n_samples, n_features)
    # For binary tree models shap_values often returns array (n_samples, n_features) or list
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # pick positive-class explanation
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary plot (dot)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_bg_trans, feature_names=feature_names, show=False)
    plt.tight_layout()
    out = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved SHAP summary -> {out}")

    # Top features by mean absolute SHAP value
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]
    top_features = [feature_names[i] for i in top_idx]

    # Dependence plots for each top feature
    for feat in top_features:
        idx = feature_names.index(feat)
        plt.figure(figsize=(6, 4))
        # shap.dependence_plot accepts either feature name or index
        shap.dependence_plot(idx, shap_vals, X_bg_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        fpath = os.path.join(output_dir, f"shap_dependence_{feat.replace('/', '_').replace(' ', '_')}.png")
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"Saved SHAP dependence -> {fpath}")

    return top_features

# -------------------------
# LIME analysis
# -------------------------
def lime_local_analysis(clf, preprocessor, X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, y_test_series: pd.Series, feature_names: List[str], output_dir: str):
    """
    Build a LimeTabularExplainer on transformed training data and explain three test cases:
    - clear positive (true label 1 with high predicted prob)
    - clear negative (true label 0 with low predicted prob)
    - borderline (pred prob closest to 0.5)
    """
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Ensure feature_names length matches
    if len(feature_names) != X_train_trans.shape[1]:
        # fallback generic names
        fname = [f"f_{i}" for i in range(X_train_trans.shape[1])]
    else:
        fname = feature_names

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train_trans),
        feature_names=fname,
        class_names=["no_default", "default"],
        mode="classification",
        random_state=0
    )

    # get predicted probabilities
    y_proba = clf.predict_proba(X_test_trans)[:, 1]

    # positive candidate: true label ==1 and high prob
    pos_idx = None
    neg_idx = None
    borderline_idx = None

    # build candidate lists
    pos_candidates = []
    neg_candidates = []
    borderline_candidates = []

    for i in range(len(y_test_series)):
        true = int(y_test_series.iloc[i])
        p = float(y_proba[i])
        if true == 1:
            pos_candidates.append((p, i))
        if true == 0:
            neg_candidates.append((p, i))
        borderline_candidates.append((abs(p - 0.5), i))

    if pos_candidates:
        pos_idx = max(pos_candidates, key=lambda x: x[0])[1]  # highest prob among true positives
    if neg_candidates:
        neg_idx = min(neg_candidates, key=lambda x: x[0])[1]  # lowest prob among true negatives
    # borderline
    borderline_idx = min(borderline_candidates, key=lambda x: x[0])[1]

    chosen = []
    for idx, label in [(pos_idx, "positive_high_prob"), (neg_idx, "negative_low_prob"), (borderline_idx, "borderline")]:
        if idx is None:
            continue
        chosen.append((idx, label))

    saved_indices = []
    for idx, label in chosen:
        instance = X_test_trans[idx]
        exp = explainer.explain_instance(instance.astype(float), clf.predict_proba, num_features=10)
        # save as matplotlib figure
        fig = exp.as_pyplot_figure()
        fname_img = os.path.join(output_dir, f"lime_{label}_{idx}.png")
        fig.savefig(fname_img, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved LIME for {label} -> {fname_img}")
        saved_indices.append(idx)

    return saved_indices

# -------------------------
# Main
# -------------------------
def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    safe_mkdir(args.output_dir)

    # 1) Load
    df = pd.read_csv(args.data)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data. Columns: {df.columns.tolist()}")

    # basic dropna on target
    df = df.dropna(subset=[args.target_col])
    df[args.target_col] = df[args.target_col].astype(int)

    # 2) Detect feature types
    numeric_features, categorical_features = detect_feature_types(df, args.target_col)
    print(f"Detected {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

    # 3) Train-test split
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(y.unique())>1 else None
    )

    # 4) Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train_raw)  # fit on train only

    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 5) Optionally apply SMOTE on training set
    if args.use_smote:
        if IMBLEARN_AVAILABLE:
            sm = SMOTE(random_state=args.random_state)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print("Applied SMOTE: new train shape:", X_train_res.shape)
        else:
            print("imblearn not available; skipping SMOTE.")
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    # 6) Create validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_res, y_train_res, test_size=0.15, random_state=args.random_state, stratify=y_train_res if len(np.unique(y_train_res))>1 else None)

    # 7) Train
    clf = train_gbm(X_tr, y_tr, X_val, y_val, args)

    # 8) Evaluate
    metrics = evaluate_model(clf, X_test, y_test)
    # save metrics and model
    joblib.dump(clf, os.path.join(args.output_dir, "model.pkl"))
    joblib.dump(preprocessor, os.path.join(args.output_dir, "preprocessor.pkl"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"auc": metrics["auc"], "precision": metrics["precision"], "recall": metrics["recall"]}, f, indent=2)

    # Save ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, np.array(metrics["y_proba"]))
    roc_auc_value = calc_auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 9) Build feature names for transformed data
    feature_names = build_feature_names(preprocessor, numeric_features, categorical_features)

    # 10) SHAP analysis (use a sample of training raw for speed)
    shap_sample = X_train_raw.sample(n=min(1000, len(X_train_raw)), random_state=args.random_state)
    try:
        top_features = shap_global_analysis(clf, preprocessor, shap_sample, feature_names, args.output_dir, top_k=args.shap_top_k)
    except Exception as e:
        print("SHAP analysis failed:", e)
        top_features = []

    # 11) LIME local explanations
    try:
        lime_cases = lime_local_analysis(clf, preprocessor, X_train_raw.sample(n=min(1000, len(X_train_raw)), random_state=args.random_state), X_test_raw, y_test.reset_index(drop=True), feature_names, args.output_dir)
    except Exception as e:
        print("LIME analysis failed:", e)
        lime_cases = []

    # 12) Short text report
    report_lines = []
    report_lines.append("Model evaluation metrics:")
    report_lines.append(f"AUC: {metrics['auc']:.4f}")
    report_lines.append(f"Precision: {metrics['precision']:.4f}")
    report_lines.append(f"Recall: {metrics['recall']:.4f}")
    report_lines.append("")
    report_lines.append("Top SHAP features:")
    for f in top_features:
        report_lines.append(f"- {f}")
    report_lines.append("")
    report_lines.append("LIME inspected test indices (positions in test set):")
    report_lines.append(", ".join([str(x) for x in lime_cases]) if lime_cases else "none")

    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("Finished. Outputs saved to:", args.output_dir)

if __name__ == "__main__":
    main()
