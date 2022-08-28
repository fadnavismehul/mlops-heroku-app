import pytest
import pandas as pd
# Testing Module
# TODO implement tests:
# 1. Check column list
# 2. Check no nulls
# 3. A test case for the GET method. This MUST test both the status code as well as the contents of the request object.
#  One test case for EACH of the possible inferences (results/outputs) of the ML model.
# 4. Test for POST

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

def test_column_names(data):

    expected_colums = [
        'age', 'workclass', 'fnlgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'salary'
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_null_presence(data):
    '''
    Check data for presence of nulls
    '''
    assert data.shape == data.dropna().shape


# TODO Add data slice test
# def test_slice_averages(data):
#     """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
#     for cat_feat in data["categorical_feat"].unique():
#         avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
#         assert (
#             2.5 > avg_value > 1.5
#         ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."


