from model.ml.data import process_data
from model.ml.model import compute_model_metrics, train_model, inference
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_process_data(data):
    
    
    X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, 
    label="salary", training=True
    )
    
    assert X.shape[0] == data.shape[0] == len(y)
    assert isinstance(encoder,OneHotEncoder)
    assert isinstance(lb,LabelBinarizer)

    
def test_compute_model_metrics_data():
    
    y = np.array([1,1,0,0])
    preds = np.array([1,0,1,0])
    
    precision, recall, fbeta = compute_model_metrics(y,preds)
    
    assert precision == 0.5
    assert recall == 0.5
    assert fbeta == 0.5
    
    
    
def test_compute_model_metrics_model(data):
    
    # Testing Model
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, 
        label="salary", training=True
    )
    
    model = train_model(X,y)
    
    assert isinstance(model,RandomForestClassifier)

    # Testing Inference
    preds = inference(model,X)
    
    assert isinstance(preds,np.ndarray)
    
    assert len(preds) == X.shape[0]
    
    # Testing Metrics

    precision, recall, fbeta = compute_model_metrics(y,preds)
    
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1