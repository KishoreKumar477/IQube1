
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

Path("reports").mkdir(exist_ok=True)

test_data = pd.read_csv("reports/test_data.csv")
X_test = test_data.drop(columns=["Attrition"])
y_test = test_data["Attrition"]

results = []

for path in Path("models").glob("*.joblib"):
    model = joblib.load(path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": path.stem,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    results.append(metrics)

    print(f"\n{path.stem}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Stayed", "Left"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {path.stem}")
    plt.tight_layout()
    plt.savefig(
        f"reports/{path.stem}_confusion_matrix.png"
    )
    plt.close()

comparison = pd.DataFrame(results)
comparison = comparison.sort_values("f1", ascending=False)

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nFinal model comparison:")
print(comparison.to_string(index=False))
