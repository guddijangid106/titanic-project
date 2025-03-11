import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Define model script names
model_scripts = [
    "titanic_logistic_regression.py",
    "titanic_random_forest.py",
    "titanic_decision_tree.py",
    "titanic_svm.py",
    "titanic_knn.py"
]

# Dictionary to store model accuracies
model_accuracies = {}

# Regex pattern to extract accuracy from script output
accuracy_pattern = re.compile(r"(\w+) Accuracy:\s*([\d\.]+)")

print("\n Running Model Scripts...\n")

for script in model_scripts:
    print(f"🔄 Running {script}...")
    try:
        # Run the script and capture output
        result = subprocess.run(["python", script], capture_output=True, text=True, check=True)
        
        # Search for accuracy in the output
        matches = accuracy_pattern.findall(result.stdout)
        if matches:
            for model_name, acc in matches:
                model_accuracies[model_name] = float(acc)
                print(f" {model_name} Accuracy: {acc}")

    except subprocess.CalledProcessError as e:
        print(f" Error running {script}: {e}")

# Convert to DataFrame
df_summary = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Accuracy"])

if not df_summary.empty:
    # Sort by Accuracy
    df_summary = df_summary.sort_values(by="Accuracy", ascending=False)

    # Find Best Model
    best_model, best_accuracy = df_summary.iloc[0]
    print(f"\n **Best Model: {best_model} with Accuracy: {best_accuracy:.4f}** 🏆\n")

    # Save summary as CSV
    summary_path = os.path.join(os.getcwd(), "titanic_model_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f" Summary saved as '{summary_path}'\n")

    # 🔹 **Plot Graph**
    plt.figure(figsize=(8, 5))
    plt.barh(df_summary["Model"], df_summary["Accuracy"], color="skyblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.title("Titanic Model Performance Comparison")
    plt.xlim(0.75, 0.85)  # Set limit for better visualization
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.gca().invert_yaxis()  # Best model at the top
    plt.show()

else:
    print("\n No accuracies found in the model scripts. Please check!\n")
