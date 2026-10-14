import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Attrition" not in df.columns:
        raise ValueError("Missing target: Attrition")

    return df

if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
    print(df.info())
    print(df["Attrition"].value_counts())