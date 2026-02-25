from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained machine learning model and scaler
model = joblib.load('model.joblib')    # Ensure model.joblib is in the same folder
scaler = joblib.load('Scaler.joblib')  # Ensure Scaler.joblib is in the same folder


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        input_data = request.get_json()

        # Required input fields
        required_fields = ["Glucose", "Blood pressure", "Body mass index", "Age"]
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract and format the data for prediction
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

        # Risk level classification
        if diabetes_risk < 30:
            level = "Low Risk"
        elif diabetes_risk < 60:
            level = "Moderate Risk"
        else:
            level = "High Risk"

        # Return the prediction
        return jsonify({
            "input": input_data,
            "Diabetes Risk (%)": round(diabetes_risk, 2),
            "Risk Level": level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port, debug=True)
