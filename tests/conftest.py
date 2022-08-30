import pytest
import pandas as pd
import pickle

@pytest.fixture(scope='session')
def data():

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = 'data/census_cleaned.csv'
    df = pd.read_csv(data_path)

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df

@pytest.fixture(scope='session')
def model():
    # Loading in model from serialized .pkl file
    pkl_filename = "rf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        rf_model, lb, encoder = pickle.load(file)
        
    return rf_model, lb, encoder
