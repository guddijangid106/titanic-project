import pandas as pd  # Import pandas for data handling

# Load the dataset
df = pd.read_csv("titanic.csv")

# Display the first 5 rows
print(df.head())

# Show general info (columns, data types, missing values)
print(df.info())

# Show missing values count
print(df.isnull().sum())
