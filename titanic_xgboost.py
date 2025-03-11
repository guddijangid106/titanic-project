import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load Data
df = pd.read_csv("titanic_processed.csv")

# ✅ Split Features and Target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ✅ Split into Train-Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", use_label_encoder=False,
    n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8
)

# ✅ Train Model
xgb_model.fit(X_train, y_train)

# ✅ Predict
y_pred = xgb_model.predict(X_test)

# ✅ Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f" XGBoost Accuracy: {accuracy:.4f}")
print(" Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Feature Importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, max_num_features=10)  # Show top 10 important features
plt.show()
