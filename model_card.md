# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- The implemented model is a RandomForest model created by Mehul Fadnavis in August 2022
- This model is used to create predictions of salary based on various features of users and classify them into two categories of "<=50K" and ">50K"
- The final model has parameters as the following:
  - n_estimators=20
  - random_state=0
- This is based on the default values given during the project starter. Additional exploration can be done to improve upon the model
- Please reach out to me for any comments or questions regarding the model
## Intended Use
- To be used for completing the module 3 assignment for the ML Devops Course

## Training & Evaluation Data
- Training dataset used is present in the `data` directory under the name `census_cleaned.csv`. The data is split into training(80%) and evalution(20%) data as per the script in `train_model.py`
## Metrics
- Precision: 0.712
- Recall: 0.621
- fbeta: 0.663
## Ethical Considerations
- The source data is a census of the US which is something to consider while using the model to make inferences in other geographies
## Caveats and Recommendations
- The model has not been checked for Bias which is something that will be picked up later
- Currently, only a RandomForest Classifier can be built, the script will need modifications to try different model types