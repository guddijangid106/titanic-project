import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Processed Titanic Data
df = pd.read_csv("titanic_processed.csv")

# ✅ Define Features (X) and Target (y)
X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target variable

# ✅ Split Data into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Initialize and Train Decision Tree Model
model = DecisionTreeClassifier(random_state=42, max_depth=5)  # You can adjust max_depth for tuning
model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f" Decision Tree Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
