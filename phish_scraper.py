import pandas as pd

# Load your dataset (corrected file path)
df = pd.read_csv(r"C:\Users\Admin\Downloads\datasets\url\phish.csv")

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (or impute them)
df = df.dropna()

# Optionally, save the cleaned dataset to a new file
df.to_csv(r"C:\Users\Admin\Downloads\datasets\url_cleaned.csv", index=False)

#2nd Display the first few rows of the dataset
print(df.head())

# Get basic statistics of numerical features
print(df.describe())

# Check the distribution of the target variable (label)
print(df["label"].value_counts())


#3rd 
# Group categories that appear less than 10 times into "Other"
threshold = 10
counts = df["TLD"].value_counts()
df["TLD"] = df["TLD"].apply(lambda x: x if counts[x] >= threshold else "Other")

from sklearn.preprocessing import MinMaxScaler

# 4th Normalize numerical features
scaler = MinMaxScaler()
numerical_features = ["URLLength", "DomainLength", "NoOfSubDomain", "NoOfLettersInURL"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

from sklearn.model_selection import train_test_split

#5th  Features (X) and labels (y)
X = df.drop("label", axis=1)  # Features
y = df["label"]  # Labels

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import pandas as pd

try:
    df = pd.read_csv("phish.csv")
    print(df.head())  # Print the first few rows to verify the data
except FileNotFoundError:
    print("Error: The file 'phish.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

