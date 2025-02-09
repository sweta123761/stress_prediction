from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "Stress Level Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        humidity = float(data["humidity"])
        temperature = float(data["temperature"])
        step_count = float(data["step_count"])

        input_data = np.array([[humidity, temperature, step_count]])
        input_data_scaled = scaler.transform(input_data)

        print("Scaled Input:", input_data_scaled)  # Debugging Line

        predicted_stress = model.predict(input_data_scaled)
        return jsonify({"stress_level": int(predicted_stress[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
