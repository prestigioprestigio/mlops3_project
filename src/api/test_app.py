import pytest
import requests
import json

@pytest.fixture
def data_true_json():
    data = {'age': 64, 'workclass': 'Self-emp-not-inc', 'fnlgt': 103643, 'education': 'Masters', 'education-num': 9, 'marital-status': 'Married-civ-spouse', 'occupation': 'Prof-speciality', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 10000, 'capital-loss': 0, 'hours-per-week': 60, 'native-country': 'United-States'}
    return data


@pytest.fixture
def data_false_json():
    data = {'age': 30, 'workclass': 'Private', 'fnlgt': 241259, 'education': 'Assoc-voc', 'education-num': 11, 'marital-status': 'Married-civ-spouse', 'occupation': 'Transport-moving', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States'}
    return data


def test_root():
    r = requests.get("http://127.0.0.1:8000/")
    assert r.status_code == 200
    assert r.json()['greeting'] == "This server is for performing inference on census data to predict whether a person's salary exceeds 50k USD"

    
def test_post_true(data_true_json):
    r = requests.post("http://127.0.0.1:8000/infer", data=json.dumps(data_true_json))
    assert r.status_code == 200
    assert r.json()['result'] == True

    
def test_post_false(data_false_json):
    r = requests.post("http://127.0.0.1:8000/infer", data=json.dumps(data_false_json))
    assert r.status_code == 200
    assert r.json()['result'] == False
    