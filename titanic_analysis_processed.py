import pandas as pd
from sklearn.preprocessing import StandardScaler

# ✅ Load Dataset
df = pd.read_csv("titanic.csv")

# ✅ Drop Unnecessary Columns
df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"], errors="ignore", inplace=True)

# ✅ Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# ✅ Convert Categorical Columns
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Convert 'Sex' to numeric
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-hot encode 'Embarked'

# ✅ Convert Boolean to Integer
df["Embarked_Q"] = df["Embarked_Q"].astype(int)
df["Embarked_S"] = df["Embarked_S"].astype(int)

# ✅ Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  

# ✅ Scale Numerical Features
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# ✅ Save Cleaned Dataset
df.to_csv("titanic_processed.csv", index=False)

print("Data Preprocessing Complete! Saved as 'titanic_processed.csv'.")
print(df.head())  
print(df.info())  # Verify all columns are numeric
