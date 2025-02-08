import numpy as np
import pandas as pd
from flask import Flask,request,jsonify, render_template,redirect,url_for
from flask_cors import CORS
import pickle
import os
#create a flask app
app = Flask(__name__, template_folder='templates/')

# Enable CORS for your Vercel frontend URL
CORS(app, origins=["https://churn-murex.vercel.app/"])  #links with the frontend 


#load the pickle model
model = pickle.load(open('ChurnModel.pkl', 'rb'))
encoder_dict = pickle.load(open('encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))  # Load the scaler

@app.route('/', methods=['GET','HEAD'])
def home():
    return render_template("index.html")
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Receive input JSON
        print("Received Data:", data)  # Debugging

        formatted_data = []

        for column, value in data.items():
            if column in encoder_dict:  # If the column needs encoding
                encoded_value = encoder_dict[column].transform([value])[0]  # Apply LabelEncoder
                formatted_data.append(encoded_value)
            elif column in ['tenure', 'MonthlyCharges', 'TotalCharges']:  # Scale numeric values
                formatted_data.append(float(value))  # Convert to float
            else:
                formatted_data.append(float(value))  # Convert other numerical values

        # Scale numerical features
        numeric_values = np.array([formatted_data[:3]])  # Extract tenure, MonthlyCharges, TotalCharges
        scaled_values = scaler.transform(numeric_values)[0]  # Apply scaling

        # Replace original values with scaled ones
        formatted_data[:3] = scaled_values.tolist()

        formatted_data = np.array([formatted_data])  # Convert to 2D array for the model
        print("Formatted Data:", formatted_data)  # Debugging

        # Make prediction
        prediction = model.predict(formatted_data)[0]  # Get single prediction

        # Convert numerical prediction to meaningful text
        result_text = "The customer will churn" if prediction == 1 else "The customer will not churn"
        print("Prediction Result:", result_text)  # Debugging

        # Redirect to result.html with prediction as a query parameter
        return redirect(url_for('result', prediction=result_text))

    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({'error': str(e)}), 400
    
@app.route('/result')
def result():
    # This route simply renders result.html with the prediction in the URL
    prediction = request.args.get('prediction')  # Get the prediction from the URL
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set
    app.run(debug=True, host="0.0.0.0", port=10000)
