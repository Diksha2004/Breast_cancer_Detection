import pickle
import numpy as np
import pandas as pd

# Load trained model, scaler, and feature names
with open("BCP.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Function to take user input for all features
def get_user_input(feature_names):
    inputs = []
    print("Enter the following features to predict breast cancer:")
    for feature in feature_names:
        value = float(input(f"{feature}: "))  # Prompt user for each feature
        inputs.append(value)
    return np.array(inputs).reshape(1, -1)  # Reshape for prediction

# Take user input
user_input = get_user_input(feature_names)

# Convert to DataFrame with feature names
user_df = pd.DataFrame(user_input, columns=feature_names)

# Scale the input using the saved scaler
user_input_scaled = scaler.transform(user_df)

# Predict
prediction = model.predict(user_input_scaled)[0]
prediction_proba = model.predict_proba(user_input_scaled)

# Print prediction probabilities for debugging
print("Prediction Probabilities:", prediction_proba)

# Map prediction to result
result = "Malignant (Cancer Detected)" if prediction == "M" else "Benign (No Cancer Detected)"

# Print final prediction
print(f"🩺 Test Prediction: {result}")

"""

# Manually provide test input values (replace with actual values)
test_input = [ 25.0, 30.0, 130.0, 1800.0, 0.20, 0.35, 0.45, 0.20, 0.35, 0.12,
    2.5, 3.5, 10.0, 250.0, 0.02, 0.05, 0.06, 0.04, 0.03, 0.007,
    30.0, 35.0, 150.0, 2500.0, 0.25, 0.50, 0.60, 0.30, 0.40, 0.15


]

# Convert test input into a DataFrame with correct feature names
test_input_df = pd.DataFrame([test_input], columns=features.columns)

# Predict using the trained model
ans = model.predict(test_input_df)

print(f"The breast cancer prediction is: {ans[0]}")



"""





