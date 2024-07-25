import requests

# Define the input data
input_data = {
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": 1,
    "person_emp_length": 6,
    "loan_intent": 3,
    "loan_amnt": 10000,
    "loan_int_rate": 10,
    "loan_percent_income": 0.2,
    "cb_person_default_on_file": 0,
    "cb_person_cred_hist_length": 5
}

# Make the POST request to the Flask API
response = requests.post('http://127.0.0.1:5000/predict', json=input_data)

# Print the response
print(response.json())
