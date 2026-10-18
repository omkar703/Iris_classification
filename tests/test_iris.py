import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import app as iris_module

@pytest.fixture
def client():
    iris_module.app.config['TESTING'] = True
    with iris_module.app.test_client() as client:
        yield client

def test_home_page(client):
    res = client.get('/')
    assert res.status_code == 200
    assert 'Iris Botanical Morphometry' in res.data.decode()

def test_predict_doc_get(client):
    res = client.get('/predict')
    assert res.status_code == 200
    data = res.get_json()
    assert 'required_features' in data
    assert 'example_request' in data

def test_predict_query_params(client):
    res = client.get('/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2')
    assert res.status_code == 200
    data = res.get_json()
    assert data['prediction'] == 0
    assert data['class_flower'] == 'setosa'

def test_predict_valid_post(client):
    payload = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    res = client.post('/predict', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['class_flower'] == 'setosa'
    assert 'probabilities' in data
    assert data['probabilities']['setosa'] > 0.9

def test_predict_aliases(client):
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.1,
        "petal_length": 4.7,
        "petal_width": 1.5
    }
    res = client.post('/predict', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data['class_flower'] == 'versicolor'

def test_predict_missing_fields(client):
    payload = {"sepal length (cm)": 5.1}
    res = client.post('/predict', json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert 'error' in data

def test_predict_invalid_data_type(client):
    payload = {
        "sepal length (cm)": "not_a_number",
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    res = client.post('/predict', json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert 'must be a valid number' in data['error']

def test_predict_out_of_bounds(client):
    payload = {
        "sepal length (cm)": -2.0,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    res = client.post('/predict', json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert 'greater than 0' in data['error']
