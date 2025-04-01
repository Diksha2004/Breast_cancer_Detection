from flask import Flask, render_template, request
from pickle import load
import numpy as np

# Load trained model and scaler
with open("BCP.pkl", "rb") as f:
    model = load(f)

with open("scaler.pkl", "rb") as f:
    scaler = load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", msg="")

@app.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input
        inputs = [
            float(request.form[feature]) for feature in [
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                "smoothness_mean", "compactness_mean", "concavity_mean",
                "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se",
                "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
                "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
            ]
        ]

        # Scale input data
        inputs = np.array([inputs]).reshape(1, -1)
        inputs_scaled = scaler.transform(inputs)  # Apply scaling

        # Make prediction
        prediction = model.predict(inputs_scaled)[0]
        result = "Malignant (Cancer Detected! you need to visit Doctor ASAP)" if prediction == "M" else "Benign (No Cancer Detected! You Are Safe!)"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

@app.route("/bra", methods=["GET"])
def bra():
    return render_template("bra.html")

if __name__ == "__main__":
    app.run(debug=True)
