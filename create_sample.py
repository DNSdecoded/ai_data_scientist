import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Survived": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    "Pclass": np.random.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5]),
    "Name": [f"Passenger_{i}" for i in range(1, n + 1)],
    "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
    "Age": np.random.normal(30, 12, n).clip(1, 80).round(1),
    "SibSp": np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
    "Parch": np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
    "Fare": np.random.exponential(30, n).round(2),
    "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]),
    "Cabin": np.random.choice(["A1", "B2", "C3", None], n, p=[0.1, 0.1, 0.1, 0.7]),
})

# Add realistic missing values
df.loc[np.random.choice(df.index, 25), "Age"] = np.nan
df.loc[np.random.choice(df.index, 15), "Fare"] = np.nan

# Make Survived correlate with features slightly
df.loc[df["Sex"] == "female", "Survived"] = np.random.choice([0, 1], (df["Sex"] == "female").sum(), p=[0.3, 0.7])
df.loc[df["Pclass"] == 1, "Survived"] = np.random.choice([0, 1], (df["Pclass"] == 1).sum(), p=[0.35, 0.65])

df.to_csv("data/titanic_sample.csv", index=False)

age_miss = df["Age"].isna().sum()
fare_miss = df["Fare"].isna().sum()
cabin_miss = df["Cabin"].isna().sum()

print(f"Created: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing: Age={age_miss}, Fare={fare_miss}, Cabin={cabin_miss}")
print(f"\nSurvival rate: {df['Survived'].mean():.1%}")
print(f"\nFirst 5 rows:")
print(df.head())
