from pathlib import Path
import joblib
import pandas as pd

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.preprocessing import prepare_features

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

RANDOM_STATE = 42

df = load_data()
X, y, _ = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Build a dedicated preprocessor for this experiment:
# median-impute numeric columns; impute and one-hot encode categorical columns.
# MaxAbsScaler is applied after the ColumnTransformer, as in the published recipe.
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = X.select_dtypes(include=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical),
    ("categorical", categorical_pipeline, categorical),
])

rfe_estimator = AdaBoostClassifier(
    n_estimators=30,
    learning_rate=0.5,
    random_state=RANDOM_STATE,
)

classifier = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=RANDOM_STATE,
)

pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("maxabs", MaxAbsScaler()),
    ("adasyn", ADASYN(random_state=RANDOM_STATE, n_neighbors=5)),
    ("rfe", RFE(
        estimator=rfe_estimator,
        n_features_to_select=0.8,
        step=0.1,
    )),
    ("classifier", classifier),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Running published-method adaptation: MaxAbsScaler + RFE(80%) + ADASYN + AdaBoost")
print("CV scoring: F1 for the attrition-positive class")
cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "average_precision": "average_precision",
    },
    n_jobs=1,
    return_train_score=False,
)

print("\n5-fold training CV (mean ± standard deviation)")
for metric in ["f1", "recall", "precision", "average_precision"]:
    values = cv_results[f"test_{metric}"]
    print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")

# Refit on the complete training split and evaluate once on the held-out test split.
pipeline.fit(X_train, y_train)
test_prob = pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\nHeld-out test results at threshold 0.50")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, test_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.4f}")
print(f"Avg prec.: {average_precision_score(y_test, test_prob):.4f}")
print(classification_report(
    y_test, test_pred, target_names=["Stayed", "Left"], zero_division=0
))

model_path = "models/AdaBoost_ADASYN_RFE.joblib"
joblib.dump(pipeline, model_path)

test_data = X_test.copy()
test_data["Attrition"] = y_test.to_numpy()
test_data.to_csv("reports/test_data_published.csv", index=False)

pd.DataFrame([{
    "model": "AdaBoost_ADASYN_RFE",
    "cv_f1_mean": cv_results["test_f1"].mean(),
    "cv_f1_std": cv_results["test_f1"].std(),
    "cv_recall_mean": cv_results["test_recall"].mean(),
    "test_accuracy": accuracy_score(y_test, test_pred),
    "test_precision": precision_score(y_test, test_pred, zero_division=0),
    "test_recall": recall_score(y_test, test_pred, zero_division=0),
    "test_f1": f1_score(y_test, test_pred, zero_division=0),
    "test_roc_auc": roc_auc_score(y_test, test_prob),
    "test_average_precision": average_precision_score(y_test, test_prob),
}]).to_csv("reports/published_pipeline_results.csv", index=False)

print(f"\nSaved model: {model_path}")
print("Saved reports/published_pipeline_results.csv")
print("Saved reports/test_data_published.csv")
