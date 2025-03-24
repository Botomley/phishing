import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load your dataset
try:
    df = pd.read_csv(r"C:\Users\Admin\Downloads\datasets\url\phish.csv")
    print("Dataset loaded successfully!")
    print(df.head())  # Display the first few rows of the dataset
except FileNotFoundError:
    print("Error: The file 'phish.csv' was not found.")
    exit()  # Stop execution if the file is not found
except Exception as e:
    print(f"An error occurred: {e}")
    exit()  # Stop execution if there is an unexpected error

# 2. Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Drop rows with missing values (or impute them)
df = df.dropna()

# 3. Check the distribution of the target variable (label)
print("Distribution of the target variable (label):")
print(df["label"].value_counts())

# 4. Handle categorical columns (e.g., "TLD")
# Group categories that appear less than 10 times into "Other"
threshold = 10
counts = df["TLD"].value_counts()
df["TLD"] = df["TLD"].apply(lambda x: x if counts[x] >= threshold else "Other")

# One-hot encode the "TLD" column
df = pd.get_dummies(df, columns=["TLD"], drop_first=True)

# 5. Normalize numerical features
scaler = MinMaxScaler()
numerical_features = ["URLLength", "DomainLength", "NoOfSubDomain", "NoOfLettersInURL"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 6. Features (X) and labels (y)
X = df.drop("label", axis=1)  # Features
y = df["label"]  # Labels

# Ensure y is numeric
le = LabelEncoder()
y = le.fit_transform(y)

# 7. Check for non-numeric columns in X
print("Data types in X:")
print(X.dtypes)

# Drop non-numeric columns (if any)
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_columns) > 0:
    print("Dropping non-numeric columns:", non_numeric_columns)
    X = X.drop(non_numeric_columns, axis=1)

# 8. Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debugging: Check shapes and data types
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Data types in X_train:")
print(X_train.dtypes)

# Check for infinite or NaN values (numeric columns only)
numeric_columns = X_train.select_dtypes(include=[np.number]).columns
print("Infinite values in X_train (numeric columns only):")
print(np.isinf(X_train[numeric_columns]).sum())
print("NaN values in X_train (numeric columns only):")
print(np.isnan(X_train[numeric_columns]).sum())

print("Dataset preprocessing completed successfully!")

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, class_weight="balanced")
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities for the positive class
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc_score)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best))


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best))