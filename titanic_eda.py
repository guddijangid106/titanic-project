import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanic_processed.csv")

# Survival count plot
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Survival by Sex
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Sex")
plt.show()

# Survival by Pclass
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution by survival
sns.histplot(df[df["Survived"] == 1]["Age"], bins=20, kde=True, color="green", label="Survived")
sns.histplot(df[df["Survived"] == 0]["Age"], bins=20, kde=True, color="red", label="Not Survived")
plt.legend()
plt.title("Age Distribution by Survival")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
