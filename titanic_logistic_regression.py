import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load the Processed Titanic Dataset
df = pd.read_csv("titanic_processed.csv")

# ✅ Define Features (X) and Target (y)
X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target

# ✅ Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence
model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f" Logistic Regression Accuracy: {accuracy:.4f}\n")

# ✅ Classification Report
print(" Classification Report:")
print(classification_report(y_test, y_pred))
