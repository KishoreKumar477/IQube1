import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data

df = load_data()

print(df.info())
print(df.isnull().sum().sort_values(ascending=False))
print("Duplicates:", df.duplicated().sum())
print(df["Attrition"].value_counts(normalize=True))

sns.countplot(data=df, x="Attrition")
plt.tight_layout()
plt.savefig("reports/attrition_distribution.png")
plt.show()

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)
plt.tight_layout()
plt.savefig("reports/overtime_attrition.png")
plt.show()