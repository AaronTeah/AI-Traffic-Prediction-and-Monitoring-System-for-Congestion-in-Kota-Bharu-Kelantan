from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model and preprocessors
model = load_model('model/traffic_lstm_model.h5')
label_encoder_day = joblib.load('model/label_encoder_day.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        logging.debug(f"Received data: {data}")

        time = pd.to_datetime(data['Time'], format='%I:%M:%S %p').time()
        time = time.hour * 3600 + time.minute * 60 + time.second

        day_of_week = label_encoder_day.transform([data['Day of the week']])[0]
        car_count = int(data['CarCount'])
        bike_count = int(data['BikeCount'])
        bus_count = int(data['BusCount'])
        truck_count = int(data['TruckCount'])

        input_data = np.array([[time, day_of_week, car_count, bike_count, bus_count, truck_count]])
        input_data = scaler.transform(input_data)
        input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))

        prediction = model.predict(input_data)
        logging.debug(f"Model prediction: {prediction}")

        predicted_total = round(float(prediction[0][0]), 0)  # Round to 2 decimal places
        logging.debug(f"Predicted total vehicles: {predicted_total}")

        return jsonify({'Predicted Total': predicted_total})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
