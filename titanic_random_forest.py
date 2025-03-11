import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Processed Data
df = pd.read_csv("titanic_processed.csv")

# ✅ Define Features (X) and Target (y)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ✅ Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f" Random Forest Accuracy: {accuracy:.4f}\n")

print(" Classification Report:")
print(classification_report(y_test, y_pred))
