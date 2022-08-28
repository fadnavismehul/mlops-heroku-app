import json
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tomlkit import string
import pickle
import pandas as pd
from model.ml.data import process_data
from fastapi.encoders import jsonable_encoder

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hi, welcome to the MLDevops Course ! This is the API for the assigment for Module 3"}


# Loading in model from serialized .pkl file
pkl_filename = "rf_model.pkl"
with open(pkl_filename, 'rb') as file:
    rf_model,lb,encoder = pickle.load(file)

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

class InferenceData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "examples": [{"age": 34, "workclass": "Private", "fnlgt": 287737, "education": "Some-college", "education-num": 10, "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Wife", "race": "White", "sex": "Female", "capital-gain": 0, "capital-loss": 1485, "hours-per-week": 40, "native-country": "United-States"}, {"age": 24, "workclass": "State-gov", "fnlgt": 123160, "education": "Masters", "education-num": 14, "marital-status": "Married-spouse-absent", "occupation": "Prof-specialty", "relationship": "Not-in-family", "race": "Asian-Pac-Islander", "sex": "Female", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 10, "native-country": "China"}, {"age": 41, "workclass": "Private", "fnlgt": 111483, "education": "HS-grad", "education-num": 9, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Own-child", "race": "White", "sex": "Male", "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}]
        }

@app.post('/predict')
async def predict(data: Union[InferenceData,List]):
    
    if isinstance(data,list):
        data = [
            InferenceData(**row) for row in data
            ]
    else:
        data = [data]

    
    # Converting input data into Pandas DataFrame
    input_df = pd.DataFrame(jsonable_encoder(data))
    processed_data,_,_,_ = process_data(input_df,
                                        categorical_features = cat_features,
                                        encoder=encoder,
                                        training=False)
    # Getting the prediction from the Logistic Regression model
    preds = rf_model.predict(processed_data)
    return {"preds":preds.tolist()}
