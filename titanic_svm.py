import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the processed dataset
df = pd.read_csv("titanic_processed.csv")

# Split data into features (X) and target (y)
X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f" SVM Accuracy: {accuracy:.4f}\n")

print(" Classification Report:")
print(classification_report(y_test, y_pred))
