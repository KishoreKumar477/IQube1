from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

from src.data_loader import load_data
from src.preprocessing import prepare_features

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

RANDOM_STATE = 42
df = load_data()
X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# SMOTE only: no class_weight balancing.
model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(
        sampling_strategy=0.5,
        k_neighbors=5,
        random_state=RANDOM_STATE,
    )),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )),
])

scoring = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "average_precision": "average_precision",
}

print("Experiment: Logistic Regression + SMOTE only")
print("class_weight=None; SMOTE sampling_strategy=0.5")
print("Five-fold stratified CV on training data only.")

cv_results = cross_validate(
    model, X_train, y_train, cv=cv, scoring=scoring,
    n_jobs=1, return_train_score=False
)
for metric in scoring:
    values = cv_results[f"test_{metric}"]
    print(f"CV {metric}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")

# OOF predictions for threshold selection. The test set is not used to select it.
oof_prob = cross_val_predict(
    model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
)[:, 1]

thresholds = np.arange(0.10, 0.801, 0.01)
threshold_rows = []
for threshold in thresholds:
    pred = (oof_prob >= threshold).astype(int)
    threshold_rows.append({
        "threshold": float(threshold),
        "precision": precision_score(y_train, pred, zero_division=0),
        "recall": recall_score(y_train, pred, zero_division=0),
        "f1": f1_score(y_train, pred, zero_division=0),
    })

threshold_df = pd.DataFrame(threshold_rows)
best_f1 = threshold_df.sort_values(
    ["f1", "precision"], ascending=False
).iloc[0]

recall_candidates = threshold_df[threshold_df["recall"] >= 0.70]
high_recall = None
if not recall_candidates.empty:
    high_recall = recall_candidates.sort_values(
        ["precision", "f1"], ascending=False
    ).iloc[0]

# Fit one final model on the complete training split.
model.fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]

results = []
operating_points = [("threshold_0.50", 0.50), ("OOF_best_F1", float(best_f1["threshold"]))]
if high_recall is not None:
    operating_points.append(("OOF_recall_at_least_0.70", float(high_recall["threshold"])))

for label, threshold in operating_points:
    test_pred = (test_prob >= threshold).astype(int)
    row = {
        "model": "LogisticRegression_SMOTE_only",
        "operating_point": label,
        "threshold": threshold,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "test_average_precision": average_precision_score(y_test, test_prob),
    }
    results.append(row)

    print(f"\n{label} (threshold={threshold:.2f})")
    print(f"Test accuracy:  {row['test_accuracy']:.4f}")
    print(f"Test precision: {row['test_precision']:.4f}")
    print(f"Test recall:    {row['test_recall']:.4f}")
    print(f"Test F1:        {row['test_f1']:.4f}")
    print(f"Test ROC-AUC:   {row['test_roc_auc']:.4f}")
    print(f"Test avg prec.: {row['test_average_precision']:.4f}")
    print(classification_report(
        y_test, test_pred, target_names=["Stayed", "Left"], zero_division=0
    ))

joblib.dump(model, "models/LogisticRegression_SMOTE_only.joblib")
pd.DataFrame(results).to_csv("reports/lr_smote_only_results.csv", index=False)
threshold_df.to_csv("reports/lr_smote_only_thresholds.csv", index=False)

print("\nSaved model: models/LogisticRegression_SMOTE_only.joblib")
print("Saved reports/lr_smote_only_results.csv")
print("Saved reports/lr_smote_only_thresholds.csv")
print("\nNote: SMOTE is currently applied after one-hot encoding. For a more")
print("methodologically sound mixed-data approach, compare against SMOTENC.")
