

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime
import numpy as  np


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB connection
#client = MongoClient('mongodb://localhost:27017/')
client = MongoClient('mongodb+srv://visheshalag03:XIbxet9QTYrQPseq@loanease.9cktikb.mongodb.net/')
loan_db = client['LoanML']  # Main LoanML database
banks_collection = loan_db['banks']  # Banks collection under LoanML database
admins_collection = loan_db['admins']  # Admins collection under LoanML database
loanAppprovalCollection = loan_db['LoanApproval']
creditRiskCollection = loan_db['CreditRisk']
LoanApproval_user_details_collection = loan_db['LoanApproval_users_details'] 
creditRisk_users_details_collection = loan_db['CreditRisk_users_details']
users_query = loan_db['user_query']

# Load data from MongoDB for loan approval prediction
cursor_loan_approval = loanAppprovalCollection.find({}, {'_id': 0})
data_loan_approval = pd.DataFrame(list(cursor_loan_approval))

# Select features for scaling for loan approval prediction
features_to_scale_loan_approval = ['income_annum', 'loan_amount', 'cibil_score',
                                   'residential_assets_value', 'commercial_assets_value',
                                   'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object for loan approval prediction
scaler_loan_approval = StandardScaler()

# Fit and transform the selected features for loan approval prediction
data_loan_approval[features_to_scale_loan_approval[:-1]] = scaler_loan_approval.fit_transform(
    data_loan_approval[features_to_scale_loan_approval[:-1]])

# Split the data for loan approval prediction into features (X) and the target variable (y)
X_loan_approval = data_loan_approval.drop(columns=['loan_status'])
y_loan_approval = data_loan_approval['loan_status']

# Apply SMOTE oversampling for loan approval prediction
smote_loan_approval = SMOTE(random_state=50)
X_resampled_loan_approval, y_resampled_loan_approval = smote_loan_approval.fit_resample(X_loan_approval, y_loan_approval)

# XGBoost classifier for loan approval prediction
xgb_classifier_loan_approval = XGBClassifier(random_state=50)
xgb_classifier_loan_approval.fit(X_resampled_loan_approval, y_resampled_loan_approval)

# Load data from MongoDB for credit risk prediction
cursor_credit_risk = creditRiskCollection.find({}, {'_id': 0})
data_credit_risk = pd.DataFrame(list(cursor_credit_risk))

# Remove the columns not needed for scaling for credit risk prediction
columns_to_scale_credit_risk = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling_credit_risk = data_credit_risk[columns_to_scale_credit_risk]

# Scale the selected columns for training data for credit risk prediction
scaler_credit_risk = StandardScaler()
X_scaled_credit_risk = scaler_credit_risk.fit_transform(X_scaling_credit_risk)
data_credit_risk[columns_to_scale_credit_risk] = X_scaled_credit_risk

# Split the training data for credit risk prediction into features and target variable
X_train_credit_risk = data_credit_risk.drop(columns=["loan_status"])
y_train_credit_risk = data_credit_risk["loan_status"]

# Apply SMOTE to balance the classes for credit risk prediction
smote_credit_risk = SMOTE(random_state=50)
X_resampled_credit_risk, y_resampled_credit_risk = smote_credit_risk.fit_resample(X_train_credit_risk, y_train_credit_risk)

# Initialize and train the XGBoost model for credit risk prediction with random_state=50
xgb_model_credit_risk = XGBClassifier(random_state=50)
xgb_model_credit_risk.fit(X_resampled_credit_risk, y_resampled_credit_risk)

def convert_floats(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    return obj


@app.route('/register_bank', methods=['POST'])
def register_bank():
    data = request.get_json()

    try:
        banks_collection.insert_one(data)
        response = {'message': 'Bank details registered successfully'}
        return jsonify(response), 200
    except Exception as e:
        response = {'message': 'Error registering bank details', 'error': str(e)}
        return jsonify(response), 500


@app.route('/bank-login', methods=['POST'])
def login_bank():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Query the bank database to check if the credentials match
    bank = banks_collection.find_one({'loginUsername': username, 'loginPassword': password})
    if bank:
        return jsonify({'message': 'Bank login successful'}), 200
    else:
        return jsonify({'message': 'Invalid bank credentials'}), 401


@app.route('/register_admin', methods=['POST'])
def register_admin():
    data = request.get_json()

    try:
        admins_collection.insert_one(data)
        response = {'message': 'Admin account created successfully'}
        return jsonify(response), 200
    except Exception as e:
        response = {'message': 'Error creating admin account', 'error': str(e)}
        return jsonify(response), 500


@app.route('/admin-login', methods=['POST'])
def login_admin():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Query the admin database to check if the credentials match
    admin = admins_collection.find_one({'username': username, 'password': password})
    if admin:
        return jsonify({'message': 'Admin login successful'}), 200
    else:
        return jsonify({'message': 'Invalid admin credentials'}), 401


# Function to preprocess user input for loan approval prediction
def preprocess_input_loan_approval(user_input):
    # Scale the specified columns
    scaled_user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df[features_to_scale_loan_approval[:-1]] = scaler_loan_approval.transform(
        scaled_user_input_df[features_to_scale_loan_approval[:-1]])
    return scaled_user_input_df


# Function to predict loan status and its probability based on user input for loan approval prediction
def predict_loan_status(user_input):
    # Preprocess user input
    scaled_input_df = preprocess_input_loan_approval(user_input)

    # Predict loan status probabilities
    loan_status_probs = xgb_classifier_loan_approval.predict_proba(scaled_input_df)[0]

    # Get the probability of approval
    probability_of_approval = loan_status_probs[1]

    # Convert probability to percentage
    probability_percentage = round(probability_of_approval * 100, 2)

    # Predict loan status
    loan_status = "Approved" if probability_of_approval >= 0.5 else "Rejected"

    return loan_status, probability_percentage


# Function to preprocess user input for credit risk prediction
def preprocess_input_credit_risk(user_input):
    # Scale the specified columns
    scaled_user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df[columns_to_scale_credit_risk] = scaler_credit_risk.transform(
        scaled_user_input_df[columns_to_scale_credit_risk])
    return scaled_user_input_df




@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    user_input = request.get_json()  # Get user input from JSON request

    # Extract user email from input data
    user_email = user_input.pop('email', None)

    # Predict loan status and probability for loan approval
    predicted_status, probability_percentage = predict_loan_status(user_input)

    # Create response JSON for loan approval
    response = {
        "Loan_Status": predicted_status,
        "Probability_of_Approval": probability_percentage
    }

    # Send email with prediction result if email is provided
    if user_email:
        send_loan_prediction_email(user_email, response)

    response = convert_floats(response)

    return jsonify(response)


def send_loan_prediction_email(user_email, prediction_result):
    try:
        # Email configuration
        sender_email = "loanease.connect@gmail.com"
        smtp_password = "hsxhxcksiauprehd"
        subject = "Loan Prediction Result"
        message_body = f"Dear User,\n\nYour loan prediction result is as follows:\n\nLoan Status: {prediction_result['Loan_Status']}\nProbability of Approval: {prediction_result['Probability_of_Approval']}%\n\nBest regards,\nLoanEase"

        # Setup the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = user_email
        msg['Subject'] = subject

        # Attach the message to the MIMEMultipart object
        msg.attach(MIMEText(message_body, 'plain'))

        # Create SMTP session for sending the mail
        with smtplib.SMTP('smtp.gmail.com', 587) as session:
            session.starttls()  # enable security
            session.login(sender_email, smtp_password)  # login with mail_id and password
            text = msg.as_string()
            session.sendmail(sender_email, user_email, text)
        
        print("Loan prediction email sent successfully.")

    except smtplib.SMTPException as e:
        print("Error sending loan prediction email:", str(e))
    except Exception as e:
        print("An unexpected error occurred while sending loan prediction email:", str(e))



#Api to fetch admins_collection
@app.route('/get_admin_by_username', methods=['GET'])
def get_admin_data():
    username = request.args.get('username')

    # Query the admin database to retrieve admin data by username
    admin = admins_collection.find_one({'username': username}, {'_id': 0})

    if admin:
        return jsonify(admin), 200
    else:
        return jsonify({'message': 'Admin not found'}), 404
    
#Api for reset password
@app.route('/change_admin_password/<username>', methods=['PATCH'])
def change_admin_password(username):
    data = request.get_json()
    new_password = data.get('new_password')

    # Update the password of the admin with the given username
    result = admins_collection.update_one({'username': username}, {'$set': {'password': new_password}})

    if result.modified_count:
        return jsonify({'message': 'Admin password updated successfully'}), 200
    else:
        return jsonify({'message': 'Admin not found'}), 404


#api to get bank data
@app.route('/get_bank_by_username', methods=['GET'])
def get_bank_data():
    username = request.args.get('username')

    # Query the bank database to retrieve bank data by username
    bank = banks_collection.find_one({'loginUsername': username}, {'_id': 0})

    if bank:
        return jsonify(bank), 200
    else:
        return jsonify({'message': 'Bank not found'}), 404
    
# Function to change bank password
# Function to change bank password
@app.route('/change_bank_password/<login_username>', methods=['PATCH'])
def change_bank_password(login_username):
    data = request.get_json()
    new_password = data.get('new_password')

    # Query the bank database to find the bank by loginUsername
    bank = banks_collection.find_one({'loginUsername': login_username})
    
    if bank:
        # Update the password
        banks_collection.update_one({'loginUsername': login_username}, {'$set': {'loginPassword': new_password}})
        return jsonify({'message': 'Bank password updated successfully'}), 200
    else:
        return jsonify({'message': 'Bank not found'}), 404 


@app.route('/store_LoanApp_user_details_with_response', methods=['POST'])
def store_LoanApp_user_details_with_response():
    data = request.get_json()

    try:
        # Store user details along with prediction response in the user_details_with_response collection
        LoanApproval_user_details_collection.insert_one(data)
        response = {'message': 'User details with prediction response stored successfully'}
        return jsonify(response), 200
    except Exception as e:
        response = {'message': 'Error storing user details with prediction response', 'error': str(e)}
        return jsonify(response), 500
    


@app.route('/store_credit_risk_user_details', methods=['POST'])
def store_credit_risk_user_details():
    data = request.get_json()

    try:
        # Store user details in the CreditRisk_users_details collection
        creditRisk_users_details_collection.insert_one(data)
        response = {'message': 'Form data stored successfully'}
        return jsonify(response), 200
    except Exception as e:
        response = {'message': 'Error storing form data', 'error': str(e)}
        return jsonify(response), 500


# Route to get all data from LoanApproval_users_details collection
@app.route('/loan_approval_user_details', methods=['GET'])
def get_loan_approval_user_details():
    # Query the LoanApproval_users_details collection to retrieve all data
    cursor = LoanApproval_user_details_collection.find({}, {'_id': 0})
    data = list(cursor)
    
    # Return the data as JSON response
    return jsonify(data)

# Route to get all data from CreditRisk_users_details collection
@app.route('/credit_risk_user_details', methods=['GET'])
def get_credit_risk_user_details():
    # Query the CreditRisk_users_details collection to retrieve all data
    cursor = creditRisk_users_details_collection.find({}, {'_id': 0})
    data = list(cursor)
    
    # Return the data as JSON response
    return jsonify(data)

@app.route('/get_all_banks', methods=['GET'])
def get_all_banks():
    # Query the banks collection to retrieve all bank data
    cursor = banks_collection.find({}, {'_id': 0})
    banks_data = list(cursor)
    
    # Return the bank data as a JSON response
    return jsonify(banks_data)

#contact us data
@app.route('/store_query', methods=['POST'])
def store_query():
    try:
        # Get the form data from the request body
        data = request.get_json()

        # Extract form data
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        # Get the current date-time in HH:mm format
        current_time = datetime.now().strftime('%H:%M')

        # Get the current date in dd-mm-yyyy format
        current_date = datetime.now().strftime('%d-%m-%Y')

        # Create a dictionary with the form data and current time
        query_data = {
            'name': name,
            'email': email,
            'subject': subject,
            'message': message,
             'date': current_date,
            'timestamp': current_time
        }

        # Insert the query data into the user_query collection
        users_query.insert_one(query_data)

        response = {'message': 'User query stored successfully'}
        return jsonify(response), 200
    except Exception as e:
        response = {'message': 'Error storing user query', 'error': str(e)}
        return jsonify(response), 500


@app.route('/get_queries', methods=['GET'])
def get_queries():
    try:
        # Fetch all documents from the collection
        query_data = users_query.find()

        # Prepare data to return
        queries = []
        for query in query_data:
            queries.append({
                'name': query['name'],
                'email': query['email'],
                'subject': query['subject'],
                'message': query['message'],
                'date': query['date'],
                'timestamp': query['timestamp']
            })

        # Return the fetched data
        return jsonify(queries), 200
    except Exception as e:
        response = {'message': 'Error fetching user queries', 'error': str(e)}
        return jsonify(response), 500

@app.route('/predict_creditrisk', methods=['POST'])
def predict_credit_risk():
    try:
        # Take input data from user (testing data) for credit risk prediction
        input_data = request.json
        
        # Print request data
        print(input_data)

        # Remove email from input data for prediction calculation
        user_email = input_data.pop('email', None)
        # Preprocess input data for credit risk prediction
        scaled_input_df = preprocess_input_credit_risk(input_data)

        # Predict using the trained model for credit risk prediction
        predictions = xgb_model_credit_risk.predict(scaled_input_df)
        probabilities = xgb_model_credit_risk.predict_proba(scaled_input_df)

        # Determine the risk range based on the probability of default
        probability_of_default = probabilities[0][1]
        if probability_of_default >= 0.7:
            risk_range = 'High Risk to Default'
        elif probability_of_default >= 0.4:
            risk_range = 'Medium Risk to Default'
        else:
            risk_range = 'Low Risk to Default'

        # Prepare response for credit risk prediction
        result = {
            "Prediction": "Person is Likely to Default" if predictions[0] == 1 else "Person is Not Likely to Default",
            "Probability of Default": round(probability_of_default * 100, 2),
            "Risk Range": risk_range
        }

        # Send email with the prediction result and user details if email is provided
        if user_email:
            send_email(user_email, result)
        result = convert_floats(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


def send_email(recipient_email, prediction_result):
    try:
        # Email configuration
        sender_email = "loanease.connect@gmail.com"
        smtp_password = "hsxhxcksiauprehd"
        subject = "Credit Risk Prediction Result"
        
        # Constructing email body with company name LoanEase
        message_body = f"Dear User,\n\nYour credit risk prediction result is as follows:\n\nPrediction: {prediction_result['Prediction']}\nProbability of Default: {prediction_result['Probability of Default']}%\nRisk Range: {prediction_result['Risk Range']}\n\nBest regards,\nLoanEase"

        # Setup the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the message to the MIMEMultipart object
        msg.attach(MIMEText(message_body, 'plain'))

        # Create SMTP session for sending the mail
        with smtplib.SMTP('smtp.gmail.com', 587) as session:
            session.starttls()  # enable security
            session.login(sender_email, smtp_password)  # login with mail_id and password
            text = msg.as_string()
            session.sendmail(sender_email, recipient_email, text)
        
        print("Email sent successfully.")

    except smtplib.SMTPException as e:
        print("Error sending email:", str(e))
    except Exception as e:
        print("An unexpected error occurred:", str(e))





#charts api's
@app.route('/plot_education_distribution', methods=['GET'])
def plot_education_distribution():
    # Count the occurrences of each education level
    education_counts = data_loan_approval['education'].value_counts().to_dict()
    return jsonify(education_counts)

@app.route('/plot_self_employed_distribution', methods=['GET'])
def plot_self_employed_distribution():
    # Count the occurrences of self-employed status
    self_employed_counts = data_loan_approval['self_employed'].value_counts().to_dict()
    return jsonify(self_employed_counts)



# Update your Flask app to include a route for calculating the approval rate
@app.route('/calculate_approval_rate', methods=['GET'])
def calculate_approval_rate():
    # Calculate the approval rate (percentage of approved applications)
    approved_count = (data_loan_approval['loan_status'] == 1).sum()
    rejected_count = (data_loan_approval['loan_status'] == 0).sum()
    total_count = len(data_loan_approval)
    approval_rate = (approved_count / total_count) * 100
    rejected_rate = (rejected_count / total_count) * 100

    # Convert values to native Python data types
    approved_count = int(approved_count)
    rejected_count = int(total_count - approved_count)
    approval_rate = float(approval_rate)
    rejected_rate = float(rejected_rate)

    # Create a dictionary with approval rate data
    approval_rate_data = {
        'Approved': approved_count,
        'Rejected': rejected_count,
        'Approval Rate': approval_rate,
        'Rejected Rate': rejected_rate
    }

    return jsonify(approval_rate_data)


@app.route('/calculate_creditrisk_rate', methods=['GET'])
def calculate_creditrisk_rate():
    # Calculate the approval rate (percentage of approved applications)
    default_count = (data_credit_risk['loan_status'] == 1).sum()
    notdefault_count = (data_credit_risk['loan_status'] == 0).sum()
    total_count = len(data_credit_risk)
    default_rate = (default_count / total_count) * 100
    notdefault_rate = (notdefault_count / total_count) * 100

    # Convert values to native Python data types
    default_count = int(default_count)
    notdefault_count = int(total_count - default_count)
    default_rate = float(default_rate)
    notdefault_rate = float(notdefault_rate)


    # Create a dictionary with approval rate data
    creditrisk_rate_data = {
        'Default': default_count,
        'Not Default': notdefault_count,
        'Default Rate': default_rate,
        'Not Default Rate': notdefault_rate
    }

    return jsonify(creditrisk_rate_data)



# Common function to calculate income distribution
def calculate_income_distribution(data, bins):
    income_distribution = np.histogram(data, bins=bins)[0]
    income_distribution_data = {
        'income_brackets': [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)],
        'customer_count': income_distribution.tolist()
    }
    return income_distribution_data

@app.route('/income_distribution_loan_approval', methods=['GET'])
def income_distribution_loan_approval():
    try:
        # Load data from MongoDB for loan approval
        cursor_loan_approval = loanAppprovalCollection.find({}, {'_id': 0, 'income_annum': 1})
        data_loan_approval = pd.DataFrame(list(cursor_loan_approval))

        # Define income brackets
        income_bins = np.arange(200000, 9900001, 500000)  # Starting from 200000, incrementing by 500000 up to 9900000
        income_bins = np.append(income_bins, 10000000)  # Append a value greater than the maximum value

        # Calculate income distribution
        income_distribution_data = calculate_income_distribution(data_loan_approval['income_annum'], income_bins)

        return jsonify(income_distribution_data), 200

    except Exception as e:
        return jsonify({'error': str(e), 'endpoint': '/income_distribution_loan_approval'}), 500

@app.route('/income_distribution_credit_risk', methods=['GET'])
def income_distribution_credit_risk():
    try:
        # Load data from MongoDB for credit risk
        cursor_credit_risk = creditRiskCollection.find({}, {'_id': 0, 'person_income': 1})
        data_credit_risk = pd.DataFrame(list(cursor_credit_risk))

        # Define income brackets
        income_bins = np.arange(4080, 223001, 20000)  # Starting from 4080, incrementing by 20000 up to 223000
        income_bins = np.append(income_bins, 250000)  # Append a value greater than the maximum value

        # Calculate income distribution
        income_distribution_data = calculate_income_distribution(data_credit_risk['person_income'], income_bins)

        return jsonify(income_distribution_data), 200

    except Exception as e:
        return jsonify({'error': str(e), 'endpoint': '/income_distribution_credit_risk'}), 500
    



@app.route('/cibil_score_approval_status', methods=['GET'])
def cibil_score_approval_status():
    try:
        # Query MongoDB collection to get data
        cursor_loan_approval = loanAppprovalCollection.find({}, {'_id': 0, 'cibil_score': 1, 'loan_status': 1})
        data_loan_approval = pd.DataFrame(list(cursor_loan_approval))

        # Define CIBIL score ranges
        cibil_score_ranges = [(300, 500), (501, 600), (601, 700), (701, 800), (801, 900)]

        # Initialize dictionary to store results
        cibil_score_approval_data = {}

        # Count number of customers in each CIBIL score range and their loan approval status
        for cibil_range in cibil_score_ranges:
            cibil_min, cibil_max = cibil_range
            filtered_data = data_loan_approval[(data_loan_approval['cibil_score'] >= cibil_min) & 
                                                (data_loan_approval['cibil_score'] <= cibil_max)]
            approval_counts = filtered_data['loan_status'].value_counts().to_dict()
            cibil_score_approval_data[f"{cibil_min}-{cibil_max}"] = {
                'Approved': approval_counts.get(1, 0),
                'Rejected': approval_counts.get(0, 0)
            }

        return jsonify(cibil_score_approval_data), 200

    except Exception as e:
        return jsonify({'error': str(e), 'endpoint': '/cibil_score_approval_status'}), 500

# Common function to calculate age distribution
def calculate_age_distribution(data, bins):
    age_distribution = np.histogram(data, bins=bins)[0]
    age_distribution_data = {
        'age_groups': [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)],
        'customer_count': age_distribution.tolist()
    }
    return age_distribution_data

@app.route('/age_distribution_credit_risk', methods=['GET'])
def age_distribution_credit_risk():
    try:
        # Load data from MongoDB for credit risk
        cursor_credit_risk = creditRiskCollection.find({}, {'_id': 0, 'person_age': 1})
        data_credit_risk = list(cursor_credit_risk)

        # Extract ages from data
        ages = [entry['person_age'] for entry in data_credit_risk]

        # Define age brackets (adjusting to match your data)
        age_bins = np.arange(20, 81, 10)  # Starting from 20, incrementing by 10 up to 80

        # Calculate age distribution
        age_distribution_data = calculate_age_distribution(ages, age_bins)

        return jsonify(age_distribution_data), 200

    except Exception as e:
        return jsonify({'error': str(e), 'endpoint': '/age_distribution_credit_risk'}), 500
    


if __name__ == '__main__':
    app.run(debug=True)