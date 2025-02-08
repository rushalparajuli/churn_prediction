import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

# Create a Flask app
app = Flask(__name__, template_folder='templates/')

# Enable CORS for Vercel frontend
CORS(app, resources={r"/predict": {"origins": "https://churn-murex.vercel.app"}})

# Load the trained model, encoders, and scaler
model = pickle.load(open('ChurnModel.pkl', 'rb'))
encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

@app.route('/', methods=['GET', 'HEAD'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        formatted_data = []
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

        for column in encoder_dict.keys():  # Encode categorical values
            if column in data:
                formatted_data.append(encoder_dict[column].transform([data[column]])[0])
            else:
                return jsonify({"error": f"Missing input: {column}"}), 400

        for column in numeric_features:  # Scale numeric values
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
        
        # Make prediction
        prediction = model.predict(formatted_data)[0]

        # Convert prediction result to text
        result_text = "The customer will churn" if prediction == 1 else "The customer will not churn"

        return jsonify({"prediction": result_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))  # Default to port 5000
    app.run(debug=True, host="0.0.0.0", port=PORT)
