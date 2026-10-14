
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET = "Attrition"

DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

def engineer_features(df):
    df = df.copy()

    df["CareerYearsOutsideCompany"] = (
        df["TotalWorkingYears"] - df["YearsAtCompany"]
    ).clip(lower=0)

    df["RoleTenureRatio"] = (
        df["YearsInCurrentRole"]
        / df["YearsAtCompany"].replace(0, 1)
    )

    df["PromotionDelayRatio"] = (
        df["YearsSinceLastPromotion"]
        / df["YearsAtCompany"].replace(0, 1)
    )

    df["ManagerTenureRatio"] = (
        df["YearsWithCurrManager"]
        / df["YearsAtCompany"].replace(0, 1)
    )

    return df

def prepare_features(df):
    df = engineer_features(df)

    y = df[TARGET].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values in Attrition")

    X = df.drop(columns=[TARGET])
    X = X.drop(
        columns=[c for c in DROP_COLUMNS if c in X.columns],
        errors="ignore"
    )

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical)
    ])

    return X, y, preprocessor

