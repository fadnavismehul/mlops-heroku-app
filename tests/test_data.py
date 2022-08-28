# Testing Module
# TODO implement tests:
# 1. Check column list
# 2. Check no nulls
# 3. A test case for the GET method. This MUST test both the status code as well as the contents of the request object.
#  One test case for EACH of the possible inferences (results/outputs) of the ML model.
# 4. Test for POST

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