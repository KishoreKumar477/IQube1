import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.data_loader import load_data
from src.preprocessing import prepare_features

RANDOM_STATE = 42
MODEL_NAME = "AdaBoost_ADASYN_RFE"

Path("reports").mkdir(exist_ok=True)

df = load_data()
X, y, _ = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model = joblib.load(f"models/{MODEL_NAME}.joblib")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# The pipeline itself contains ADASYN, so resampling is refitted only within
# each training fold by cross_val_predict.
oof_prob = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

thresholds = np.arange(0.10, 0.801, 0.01)
rows = []

for threshold in thresholds:
    pred = (oof_prob >= threshold).astype(int)
    rows.append({
        "threshold": threshold,
        "precision": precision_score(y_train, pred, zero_division=0),
        "recall": recall_score(y_train, pred, zero_division=0),
        "f1": f1_score(y_train, pred, zero_division=0),
    })

threshold_table = pd.DataFrame(rows)
best_f1_row = threshold_table.loc[threshold_table["f1"].idxmax()]

# Also identify the highest threshold that reaches a chosen recall target.
# This is a training-only target; the test set is not used to select it.
recall_target = 0.75
eligible = threshold_table[threshold_table["recall"] >= recall_target]
target_recall_row = eligible.sort_values(
    ["precision", "threshold"], ascending=[False, False]
).head(1)

chosen_rows = [("Best OOF F1", best_f1_row)]
if not target_recall_row.empty:
    chosen_rows.append(("Recall >= 0.75, highest precision", target_recall_row.iloc[0]))

# Fit the model on all training data before evaluating the chosen thresholds.
# The threshold is selected only from training OOF predictions.
model.fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]

for label, row in chosen_rows:
    threshold = float(row["threshold"])
    test_pred = (test_prob >= threshold).astype(int)

    print(f"\n{MODEL_NAME} — {label}")
    print(f"Threshold selected from training OOF predictions: {threshold:.2f}")
    print(f"OOF precision: {row['precision']:.4f}")
    print(f"OOF recall:    {row['recall']:.4f}")
    print(f"OOF F1:        {row['f1']:.4f}")
    print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test F1:        {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test average precision: {average_precision_score(y_test, test_prob):.4f}")
    print(classification_report(
        y_test, test_pred, target_names=["Stayed", "Left"], zero_division=0
    ))

threshold_table.to_csv("reports/AdaBoost_ADASYN_RFE_thresholds.csv", index=False)

plt.figure()
plt.plot(threshold_table["threshold"], threshold_table["f1"], label="OOF F1")
plt.plot(threshold_table["threshold"], threshold_table["recall"], label="OOF recall")
plt.axvline(float(best_f1_row["threshold"]), linestyle="--",
            label=f"Best F1 threshold: {best_f1_row['threshold']:.2f}")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("AdaBoost + ADASYN + RFE: threshold selection")
plt.legend()
plt.tight_layout()
plt.savefig("reports/AdaBoost_ADASYN_RFE_threshold_curve.png")
plt.close()

print("\nSaved reports/AdaBoost_ADASYN_RFE_thresholds.csv")
print("Saved reports/AdaBoost_ADASYN_RFE_threshold_curve.png")
