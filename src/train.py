
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.preprocessing import prepare_features

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

df = load_data()
X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        #class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    results.append({
        "model": name,
        **{
            metric: scores[f"test_{metric}"].mean()
            for metric in [
                "accuracy", "precision", "recall",
                "f1", "roc_auc"
            ]
        }
    })

    print(f"\n{name}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{metric}: {scores[f'test_{metric}'].mean():.4f}")

pd.DataFrame(results).to_csv(
    "reports/baseline_cv_results.csv",
    index=False
)

print("\nSaved baseline results.")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
