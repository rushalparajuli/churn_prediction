import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

# Create a Flask app
app = Flask(__name__, template_folder='templates/')

# Enable CORS for all routes
CORS(app)

# Load the trained model, encoders, and scaler
try:
    model = pickle.load(open('ChurnModel.pkl', 'rb'))
    encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    print("Model, encoders, and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model, encoders, or scaler: {e}")

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging: Log received data

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        formatted_data = []
        numeric_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']

        # Encode categorical values
        for column in encoder_dict.keys():
            if column in data:
                formatted_data.append(encoder_dict[column].transform([data[column]])[0])
            else:
                return jsonify({"error": f"Missing input: {column}"}), 400

        # Scale numeric values
        for column in numeric_features:
            if column in data:
                formatted_data.append(float(data[column]))  # Convert to float
            else:
                return jsonify({"error": f"Missing numeric input: {column}"}), 400

        # Scale numeric values
        numeric_values = np.array([formatted_data[-3:]])  # Extract last three (numeric) values
        scaled_values = scaler.transform(numeric_values)[0]  # Apply scaling
        formatted_data[-3:] = scaled_values.tolist()  # Replace with scaled values

        # Convert to 2D array for model prediction
        formatted_data = np.array([formatted_data])
        print("Formatted data for prediction:", formatted_data)  # Debugging: Log formatted data

        # Make prediction
        prediction = model.predict(formatted_data)[0]
        print("Prediction:", prediction)  # Debugging: Log prediction

        # Convert prediction result to text
        result_text = "The customer will churn" if prediction == 1 else "The customer will not churn"

        return jsonify({"prediction": result_text})

    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging: Log errors
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))  # Default to port 5000
    app.run(debug=True, host="0.0.0.0", port=PORT)