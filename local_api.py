import json

import requests

# DONE: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# DONE: print the status code
# DONE: print the welcome message
print(f"Status Code: {r.status_code}")
print(f"Welcome Message: {r.json()}")




data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# DONE: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data)

# DONE: print the status code
# DONE: print the result
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()}")
