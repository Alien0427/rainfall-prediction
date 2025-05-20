from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import os
from database import Database
import requests
from functools import wraps
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
db = Database()

# Load the trained model
model_path = 'models/rainfall_model_random_forest.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    if os.environ.get('FLASK_ENV') == 'testing':
        # Dummy model for testing
        class DummyModel:
            def predict(self, X):
                return [0 for _ in range(len(X))]
            def predict_proba(self, X):
                return [[0.5, 0.5] for _ in range(len(X))]
        model = DummyModel()
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def preprocess_input(data):
    """Preprocess input data to match model requirements"""
    # Convert input data to numpy array
    features = np.array([
        float(data['temperature']),
        float(data['humidity']),
        float(data['wind_speed']),
        float(data['pressure'])
    ]).reshape(1, -1)
    return features

def get_weather_data(latitude, longitude):
    """Get current weather data from OpenWeatherMap API"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return None
        
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    
    try:
        response = requests.get(url)
        data = response.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'pressure': data['main']['pressure']
        }
    except Exception as e:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        location = data.get('location', 'Unknown')
        
        # If location is provided, get current weather data
        if location != 'Unknown':
            locations = db.get_locations()
            location_data = next((loc for loc in locations if loc['name'] == location), None)
            if location_data:
                weather_data = get_weather_data(location_data['latitude'], location_data['longitude'])
                if weather_data:
                    data.update(weather_data)
        
        # Preprocess the input data
        features = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        result = {
            'prediction': bool(prediction),
            'probability': float(probability),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': location
        }
        
        # Save prediction to database
        db.save_prediction({
            **data,
            'prediction': prediction,
            'probability': probability,
            'location': location
        })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history')
def get_history():
    predictions = db.get_recent_predictions()
    return jsonify(predictions)

@app.route('/locations')
def get_locations():
    locations = db.get_locations()
    return jsonify(locations)

@app.route('/add_location', methods=['POST'])
def add_location():
    try:
        data = request.get_json()
        success = db.add_location(
            data['name'],
            float(data['latitude']),
            float(data['longitude'])
        )
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 