from model.ml.model import compute_model_metrics, inference
from model.ml.data import process_data
from typing import List
import pandas as pd
import pickle

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

def get_metrics_on_slice(column: str,data_path: str) -> List:
    
    df = pd.read_csv(data_path)

    # Loading in model from serialized .pkl file
    pkl_filename = "rf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        rf_model, lb, encoder = pickle.load(file)
    
    X, y, _,_ = process_data(
        df, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb
    )
    preds = inference(rf_model, X)

    output_list = []
    for slice_val in df[column].unique():
        idx = df[df[column] == slice_val].index
        preds_slice = preds[idx]
        y_slice = y[idx]
        precision, recall, fbeta = compute_model_metrics(y_slice,preds_slice)
        dict_output = {
            "slice_value": slice_val,
            "Precision": precision,
            "Recall": recall,
            "fbeta": fbeta
        }
        output_list.append(dict_output)
    return output_list

if __name__ == "__main__":
    metric_slices =  get_metrics_on_slice(column="education",data_path="data/census_cleaned.csv")
    with open('slice_output.txt','w+') as f:
        for slice in metric_slices:
            f.write(str(slice))
            f.write('\n')