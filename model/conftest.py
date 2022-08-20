import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data(request):

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = './data/cleaned_census.csv'

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df