import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

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