from fastapi.testclient import TestClient
from main import app
import json
import pandas as pd

client = TestClient(app)


def test_say_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "greeting": "Hi, welcome to the MLDevops Course ! This is the API for the assigment for Module 3"}


def test_predict():
    with open('sample_data.csv') as f:
        df = pd.read_csv(f).drop(['Unnamed: 0'], axis=1)
    data = json.dumps(df.to_dict(orient='rows'))
    response = client.post("/predict", data=data)
    assert response.status_code == 200


def test_predict_one():
    single_sample = {
        "age": 34,
        "workclass": "Private",
        "fnlgt": 287737,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1485,
        "hours-per-week": 40,
        "native-country": "United-States"}
    data = json.dumps(single_sample)
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert response.json() == {"preds": [1]}


def test_predict_zero():
    single_sample = {
        "age": 24,
        "workclass": "State-gov",
        "fnlgt": 123160,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-spouse-absent",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 10,
        "native-country": "China"}
    data = json.dumps(single_sample)
    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert response.json() == {"preds": [0]}
