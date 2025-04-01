import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
data = pd.read_csv("data.csv")

# Standardize column names (lowercase and replace spaces with underscores)
data.columns = data.columns.str.lower().str.replace(" ", "_")

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Features and target
X = data.drop("diagnosis", axis=1)  # Drop target column
y = data["diagnosis"]  # Target variable

# Print feature names for debugging
print("Feature names in training data:", list(X.columns))

# Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print class distribution before and after SMOTE
print("Class distribution before SMOTE:\n", y.value_counts())
print("Class distribution after SMOTE:\n", y_resampled.value_counts())

# Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print(f"Number of features: {x_train.shape[1]}")

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Print mean and standard deviation of scaled data for debugging
print("Mean of scaled training data:", np.mean(x_train, axis=0))
print("Standard deviation of scaled training data:", np.std(x_train, axis=0))

# Model (Random Forest to avoid overfitting)
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=6,  # Maximum depth of each tree
    random_state=42  # For reproducibility
)
model.fit(x_train, y_train)

# Evaluate Model
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, model.predict(x_test))

print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Testing Accuracy: {test_accuracy:.2%}")

# Confusion Matrix
y_pred = model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Prediction Probabilities (for debugging)
y_pred_proba = model.predict_proba(x_test)
print("Prediction Probabilities for Test Set:\n", y_pred_proba[:5])  # Print first 5 samples

# Save model, scaler, and feature names
with open("BCP.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model, Scaler, and Feature Names saved successfully as BCP.pkl, scaler.pkl, and feature_names.pkl")