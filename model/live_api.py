import requests
import json
response = requests.get('https://mldevops-test.herokuapp.com/')

print(response.content.decode())

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
response = requests.post('https://mldevops-test.herokuapp.com/predict',
                        data=data)


print(response.content.decode())
