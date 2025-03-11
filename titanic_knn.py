import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Processed Titanic Data
df = pd.read_csv("titanic_processed.csv")

# ✅ Split Features and Target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ✅ Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize Features (KNN is sensitive to scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train KNN Classifier (default: k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ✅ Predictions & Evaluation
y_pred = knn.predict(X_test)

# ✅ Print Accuracy & Classification Report
accuracy = accuracy_score(y_test, y_pred)
print(f" KNN Accuracy: {accuracy:.4f}\n")
print(" Classification Report:")
print(classification_report(y_test, y_pred))
