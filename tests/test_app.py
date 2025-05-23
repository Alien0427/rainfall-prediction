import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pytest
from app import app as flask_app
import json

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    # Set up test environment variables
    os.environ['OPENWEATHER_API_KEY'] = 'test_key'
    os.environ['FLASK_ENV'] = 'testing'
    with flask_app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Rainfall Prediction' in response.data

def test_predict_endpoint(client):
    test_data = {
        'temperature': 25.0,
        'humidity': 60.0,
        'wind_speed': 10.0,
        'pressure': 1013.0
    }
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert 'timestamp' in data

def test_predict_endpoint_invalid_data(client):
    test_data = {
        'temperature': 'invalid',
        'humidity': 60.0,
        'wind_speed': 10.0,
        'pressure': 1013.0
    }
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 400

def test_history_endpoint(client):
    response = client.get('/history')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_locations_endpoint(client):
    response = client.get('/locations')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)

def test_add_location_endpoint(client):
    test_location = {
        'name': 'Test City',
        'latitude': 0.0,
        'longitude': 0.0
    }
    response = client.post('/add_location',
                          data=json.dumps(test_location),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'success' in data

def test_add_location_endpoint_invalid_data(client):
    test_location = {
        'name': 'Test City',
        'latitude': 'invalid',
        'longitude': 0.0
    }
    response = client.post('/add_location',
                          data=json.dumps(test_location),
                          content_type='application/json')
    assert response.status_code == 400 