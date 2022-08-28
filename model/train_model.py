# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from model.ml.data import process_data
from model.ml.model import train_model, compute_model_metrics, inference

import pickle
import pandas as pd

# Add code to load in the data.

data = pd.read_csv('./data/census_cleaned.csv')


# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


print(X_train.shape)
# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

trained_model = train_model(X_train, y_train)

pkl_filename = "rf_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump([trained_model, lb, encoder], file)


preds = inference(trained_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)

result_str = f'''
Precision: {precision}
Recall: {recall}
fbeta: {fbeta}
'''

print(result_str)
