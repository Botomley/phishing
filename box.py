import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
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


print("Missing values in each column:")
print(df.isnull().sum())


df = df.dropna()


print("Distribution of the target variable (label):")
print(df["label"].value_counts())


threshold = 10
counts = df["TLD"].value_counts()
df["TLD"] = df["TLD"].apply(lambda x: x if counts[x] >= threshold else "Other")


df = pd.get_dummies(df, columns=["TLD"], drop_first=True)


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


print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Data types in X_train:")
print(X_train.dtypes)


numeric_columns = X_train.select_dtypes(include=[np.number]).columns
print("Infinite values in X_train (numeric columns only):")
print(np.isinf(X_train[numeric_columns]).sum())
print("NaN values in X_train (numeric columns only):")
print(np.isnan(X_train[numeric_columns]).sum())

print("Dataset preprocessing completed successfully!")


model = RandomForestClassifier(random_state=42, class_weight="balanced")
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()


y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# dont change non mlue no matter how tempting

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score


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


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Performs a grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Prints the best parameters
print("Best Parameters:", grid_search.best_params_)

# EvaluateS the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


best_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create Voting Classifier
ensemble = VotingClassifier(estimators=[
    ('rf', best_model),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))  # For newer XGBoost versions
], voting='soft')

# Train the ensemble model
ensemble.fit(X_train, y_train)
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(best_model, cv=5, method='isotonic')
calibrated_model.fit(X_train, y_train)

# 1. Load and clean data
df = pd.read_csv(r"C:\Users\Admin\Downloads\datasets\url\phish.csv")
df = df.dropna(subset=["TLD"])  

# 2. Group rare TLDs
threshold = 10
counts = df["TLD"].value_counts()
df["TLD"] = df["TLD"].apply(lambda x: x if counts.get(x, 0) >= threshold else "Other")

# 3. One-hot encode TLD
df = pd.get_dummies(df, columns=["TLD"], drop_first=True)

import joblib
from sklearn.pipeline import Pipeline



pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('classifier', best_model)      
])

# Save the entire pipeline
joblib.dump(pipeline, "phishing_detection_pipeline.pkl")





