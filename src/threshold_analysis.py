
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    classification_report
)

from src.data_loader import load_data
from src.preprocessing import prepare_features

df = load_data()
X, y, _ = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
    model = joblib.load(f"models/{name}.joblib")

    # Generate probabilities without training on each row's fold.
    oof_prob = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    thresholds = np.arange(0.10, 0.71, 0.01)

    scores = [
        f1_score(
            y_train,
            (oof_prob >= threshold).astype(int)
        )
        for threshold in thresholds
    ]

    best_idx = int(np.argmax(scores))
    best_threshold = thresholds[best_idx]

    # Final evaluation uses the untouched test set.
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    print(f"\n{name}")
    print(f"OOF-selected threshold: {best_threshold:.2f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test recall: {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test F1: {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test average precision: {average_precision_score(y_test, test_prob):.4f}")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["Stayed", "Left"],
        zero_division=0
    ))

    plt.figure()
    plt.plot(thresholds, scores)
    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Selected: {best_threshold:.2f}"
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Training out-of-fold F1")
    plt.title(f"{name}: threshold selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reports/{name}_threshold_curve.png")
    plt.close()
