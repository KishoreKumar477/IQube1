
from pathlib import Path
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.preprocessing import prepare_features

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

df = load_data()
X, y, preprocessor = prepare_features(df)

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

searches = {
    "LogisticRegression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        {
            "classifier__C": [0.01, 0.1, 1, 10]
        },
        "grid"
    ),

    "RandomForest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 8, 12, 20],
            "classifier__min_samples_leaf": [2, 4, 6],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__max_features": ["sqrt", 0.8],
            "classifier__class_weight": [
                None, "balanced", "balanced_subsample"
            ]
        },
        "random"
    ),

    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
            "classifier__scale_pos_weight": [1, 3, 5]
        },
        "random"
    )
}

results = []

for name, (classifier, params, search_type) in searches.items():
    print(f"\nStarting {name}...", flush=True)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(
            sampling_strategy=0.5,
            k_neighbors=5,
            random_state=42
        )),
        ("classifier", classifier)
    ])
    
    print("SMOTE ENABLED:", "smote" in pipeline.named_steps)
    print(pipeline)

    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="f1",
            cv=cv,
            n_jobs=4,
            refit=True,
            verbose=1
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=20,
            scoring="f1",
            cv=cv,
            n_jobs=4,
            random_state=42,
            refit=True,
            verbose=1
        )

    search.fit(X_train, y_train)

    joblib.dump(
        search.best_estimator_,
        f"models/{name}.joblib"
    )

    results.append({
        "model": name,
        "best_cv_f1": search.best_score_,
        "best_params": str(search.best_params_)
    })

    print(f"Completed {name}", flush=True)
    print("Best CV F1:", round(search.best_score_, 4))
    print("Best parameters:", search.best_params_)

pd.DataFrame(results).to_csv(
    "reports/tuning_results.csv",
    index=False
)

test_data = X_test.copy()
test_data["Attrition"] = y_test.to_numpy()

test_data.to_csv(
    "reports/test_data.csv",
    index=False
)

print("\nTuning completed. Results saved.", flush=True)
