from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

@app.route('/')
def index():
    return "Diabetes Risk Prediction API is running"

@app.route('/hanuman/<name>')
def hanuman(name: str):
    return f"{name} is a hanuman"

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get JSON input
        input_data = request.get_json()

        # Extract features (ORDER IS IMPORTANT)
        data = np.array([
            input_data["Glucose"],
            input_data["Blood pressure"],
            input_data["Body mass index"],
            input_data["Age"]
        ]).reshape(1, -1)

        # Apply standardization
        data = scaler.transform(data)

        # Predict probability
        probability = model.predict_proba(data)
        diabetes_risk = probability[0][1] * 100

        # Risk level
        if diabetes_risk < 30:
            level = "Low Risk"
        elif diabetes_risk < 60:
            level = "Moderate Risk"
        else:
            level = "High Risk"

        return jsonify({
            "input": input_data,
            "Diabetes Risk (%)": round(diabetes_risk, 2),
            "Risk Level": level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
