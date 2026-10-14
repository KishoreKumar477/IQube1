from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler, SMOTENC, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data
from src.preprocessing import engineer_features, TARGET, DROP_COLUMNS

RANDOM_STATE = 42
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def prepare_raw_features(df):
    """Feature engineering + target extraction, leaving categorical columns unencoded."""
    df = engineer_features(df.copy())
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected values in Attrition")
    X = df.drop(columns=[TARGET])
    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns], errors="ignore")
    cat_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return X, y, num_cols, cat_cols


def make_preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, num_cols),
        ("categorical", categorical_pipe, cat_cols),
    ])


df = load_data()
X, y, num_cols, cat_cols = prepare_raw_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"Raw feature columns: {X.shape[1]}")
print(f"Numerical columns: {len(num_cols)} | categorical columns: {len(cat_cols)}")
print("Models: LR + SMOTENC only; LR + RandomOverSampler only; LR + ordinary SMOTE reference")
print("All models use class_weight=None and the same split/folds.")

# SMOTENC works on mixed numeric/categorical data before one-hot encoding.
# Scale numeric columns before sampling so distance calculations are meaningful,
# while categorical columns remain strings and are identified by column name.
pre_sampler_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("categorical", SimpleImputer(strategy="most_frequent"), cat_cols),
], verbose_feature_names_out=False).set_output(transform="pandas")

post_sampler_preprocessor = ColumnTransformer([
    ("numeric", "passthrough", num_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

smotenc_model = ImbPipeline([
    ("pre_sampler_preprocessor", pre_sampler_preprocessor),
    ("smotenc", SMOTENC(
        categorical_features=cat_cols,
        sampling_strategy=0.5,
        k_neighbors=5,
        random_state=RANDOM_STATE,
    )),
    ("post_sampler_preprocessor", post_sampler_preprocessor),
    ("classifier", LogisticRegression(
        C=1.0, max_iter=3000, class_weight=None, random_state=RANDOM_STATE
    )),
])

# Random oversampling uses the same original raw columns; resampling occurs
# before one-hot encoding, so no synthetic fractional categorical values.
ros_model = ImbPipeline([
    ("pre_sampler_preprocessor", pre_sampler_preprocessor),
    ("sampler", RandomOverSampler(
        sampling_strategy=0.5, random_state=RANDOM_STATE
    )),
    ("post_sampler_preprocessor", post_sampler_preprocessor),
    ("classifier", LogisticRegression(
        C=1.0, max_iter=3000, class_weight=None, random_state=RANDOM_STATE
    )),
])

# Ordinary SMOTE reference matching the previous experiment (one-hot before SMOTE).
ordinary_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])
ordinary_smote_model = ImbPipeline([
    ("preprocessor", ordinary_preprocessor),
    ("smote", SMOTE(
        sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE
    )),
    ("classifier", LogisticRegression(
        C=1.0, max_iter=3000, class_weight=None, random_state=RANDOM_STATE
    )),
])

models = {
    "LR_SMOTENC_only": smotenc_model,
    "LR_RandomOverSampler_only": ros_model,
    "LR_ordinary_SMOTE_reference": ordinary_smote_model,
}

scoring = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "average_precision": "average_precision",
}

summary = []
for name, model in models.items():
    print(f"\n{'=' * 70}\n{name}", flush=True)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=1, return_train_score=False
    )
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        print(f"CV {metric}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

    oof_prob = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    thresholds = np.arange(0.10, 0.801, 0.01)
    thr_rows = []
    for t in thresholds:
        pred = (oof_prob >= t).astype(int)
        thr_rows.append({
            "threshold": float(t),
            "precision": precision_score(y_train, pred, zero_division=0),
            "recall": recall_score(y_train, pred, zero_division=0),
            "f1": f1_score(y_train, pred, zero_division=0),
        })
    thr_df = pd.DataFrame(thr_rows)
    best_f1 = thr_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
    recall_candidates = thr_df[thr_df["recall"] >= 0.70]
    best_recall = None
    if not recall_candidates.empty:
        best_recall = recall_candidates.sort_values(
            ["precision", "f1"], ascending=False
        ).iloc[0]

    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)[:, 1]
    operating_points = [
        ("threshold_0.50", 0.50),
        ("OOF_best_F1", float(best_f1["threshold"])),
    ]
    if best_recall is not None:
        operating_points.append(("OOF_recall_at_least_0.70", float(best_recall["threshold"])))

    for point, threshold in operating_points:
        pred = (test_prob >= threshold).astype(int)
        row = {
            "model": name,
            "operating_point": point,
            "threshold": threshold,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, test_prob),
            "test_average_precision": average_precision_score(y_test, test_prob),
            "oof_precision": float(best_f1["precision"]) if point == "OOF_best_F1" else None,
            "oof_recall": float(best_f1["recall"]) if point == "OOF_best_F1" else None,
            "oof_f1": float(best_f1["f1"]) if point == "OOF_best_F1" else None,
        }
        summary.append(row)
        print(f"\n{point} | threshold={threshold:.2f}")
        print(
            f"TEST accuracy={row['test_accuracy']:.4f} | "
            f"precision={row['test_precision']:.4f} | recall={row['test_recall']:.4f} | "
            f"F1={row['test_f1']:.4f} | ROC-AUC={row['test_roc_auc']:.4f} | "
            f"AP={row['test_average_precision']:.4f}"
        )
        print(classification_report(
            y_test, pred, target_names=["Stayed", "Left"], zero_division=0
        ))

    # Persist all candidates separately; the existing champion file is untouched.
    joblib.dump(model, f"models/{name}.joblib")
    thr_df.insert(0, "model", name)
    thr_df.to_csv(f"reports/{name}_thresholds.csv", index=False)

pd.DataFrame(summary).to_csv("reports/sampler_comparison.csv", index=False)
print("\nSaved reports/sampler_comparison.csv")
print("Saved per-model threshold tables in reports/")
print("Saved candidate models in models/ (existing champion model untouched).")
print("\nImportant: choose based on CV stability plus held-out test metrics;")
print("the test set has only 47 positive attrition cases.")
