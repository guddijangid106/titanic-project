import pandas as pd
from sklearn.preprocessing import StandardScaler

# ✅ Load Dataset
df = pd.read_csv("titanic.csv")

# ✅ Extract Titles from Names
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

# ✅ Group Rare Titles
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Countess": "Rare", "Don": "Rare",
    "Dona": "Rare", "Jonkheer": "Rare", "Lady": "Rare", "Sir": "Rare", "Capt": "Rare"
}
df["Title"] = df["Title"].map(title_mapping)

# ✅ Convert Categorical Columns
df = pd.get_dummies(df, columns=["Title"], drop_first=True)  # One-hot encode Titles

# ✅ Drop Unnecessary Columns
df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], inplace=True)

# ✅ Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# ✅ Encode 'Sex' and 'Embarked'
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# ✅ Feature Engineering: Create Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  

# ✅ Scale Numerical Features
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# ✅ Save Cleaned Dataset
df.to_csv("titanic_processed.csv", index=False)

print("✅ Data Preprocessing Updated! Extracted Titles & Saved as 'titanic_processed.csv'.")
print(df.head())  
print(df.info())  # Verify all columns are numeric
