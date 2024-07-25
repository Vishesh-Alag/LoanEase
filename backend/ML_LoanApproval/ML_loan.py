#demo
'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = pd.read_csv("LoanEase_DataCleaning/loandata.csv")

# Remove leading/trailing whitespaces in column names
data.columns = data.columns.str.strip()

# Perform label encoding
label_mapping = {
    "education": {" Graduate": 1, " Not Graduate": 0},
    "self_employed": {" Yes": 1, " No": 0},
    "loan_status": {" Approved": 1, " Rejected": 0}
}
data.replace(label_mapping, inplace=True)


numeric_features = ['income_annum', 'loan_amount','cibil_score' ,
                     'residential_assets_value', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value']


# Outlier Detection
# Visualize outliers using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_features])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# Detect outliers using Z-score method
z_scores = np.abs(stats.zscore(data[numeric_features]))
outliers = (z_scores > 3).any(axis=1)

# Remove outliers
data = data[~outliers]

print("Data after removing outliers:")

# we have 4166 rows of data now findout majority and miniority of our target variable class
print((data['loan_status']==1).sum())
print((data['loan_status']==0).sum())

df=pd.DataFrame(data)'''
#df.to_csv('LoanMLData.csv',index=False)







#USING SMOTE FOR HANDLING OVERSAMPLING

'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv("LoanEase_DataCleaning/loandata.csv")

# Remove leading/trailing whitespaces in column names
data.columns = data.columns.str.strip()

# Perform label encoding
label_mapping = {
    "education": {" Graduate": 1, " Not Graduate": 0},
    "self_employed": {" Yes": 1, " No": 0},
    "loan_status": {" Approved": 1, " Rejected": 0}
}
data.replace(label_mapping, inplace=True)

# Feature Scaling (z-score normalization)
scaler = StandardScaler()
numeric_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
                    'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value']

data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Outlier Detection
# Visualize outliers using box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_features])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

# Detect outliers using Z-score method
z_scores = np.abs(stats.zscore(data[numeric_features]))
outliers = (z_scores > 3).any(axis=1)

# Remove outliers
data = data[~outliers]

print("Data after removing outliers:")
#print(data)

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier()
}

# Train and evaluate classifiers
for name, classifier in classifiers.items():
    print(f"\n\nClassifier: {name}")
    classifier.fit(X_train, y_train)
    
    # Training data prediction and evaluation
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training Data Accuracy:", train_accuracy * 100)
    
    # Testing data prediction and evaluation
    y_test_pred = classifier.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("\nTesting Data Accuracy:", test_accuracy * 100)'''











#RANDOM FOREST USING SMOTE

'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMldata.csv", index_col=None)

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Training data prediction and evaluation
y_train_pred = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Data Accuracy:", train_accuracy * 100)

# Testing data prediction and evaluation
y_test_pred = rf_classifier.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTesting Data Accuracy:", test_accuracy * 100)

# Feature Importance
feature_importance = rf_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=X.columns[sorted_idx])
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()'''




#XGB CLASSIFIER USING SMOTE
'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMlData.csv", index_col=None)

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

# Training data prediction and evaluation
y_train_pred = xgb_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Data Accuracy:", train_accuracy * 100)

# Testing data prediction and evaluation
y_test_pred = xgb_classifier.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
classification_rep = classification_report(y_test, y_test_pred)
print(classification_rep)

# Extracting F1 score from classification report
f1 = f1_score(y_test, y_test_pred)
print("F1 Score:", f1)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTesting Data Accuracy:", test_accuracy * 100)

# Feature Importance
feature_importance = xgb_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=X.columns[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()'''















#------------------------------------------------------------------------------------


#XGB CLASSIFIER USING SMOTE
'''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the data without setting the index column
data = pd.read_csv("LoanMlData.csv", index_col=None)

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_resampled, y_resampled)

# Training data prediction and evaluation
y_pred = xgb_classifier.predict(X_resampled)
train_accuracy = accuracy_score(y_resampled, y_pred)
print("Training Data Accuracy:", train_accuracy * 100)

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_resampled, y_pred))
print("\nClassification Report:")
classification_rep = classification_report(y_resampled, y_pred)
print(classification_rep)

# Extracting F1 score from classification report
f1 = f1_score(y_resampled, y_pred)
print("F1 Score:", f1)

# Feature Importance
feature_importance = xgb_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=X.columns[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Save the trained model to disk
model_filename = "xgboost_model.pkl"
joblib.dump(xgb_classifier, model_filename)
print("Trained model saved successfully as:", model_filename)'''





#-------------WITHOUT NORMALISED DATASET





# with feature scaling, #compariosn between different models
'''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMLData.csv", index_col=None)

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the specified features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=50)

# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=50),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=50),
    "XGBoost": XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=50),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate classifiers
for name, classifier in classifiers.items():
    print(f"\n\nClassifier: {name}")
    classifier.fit(X_train, y_train)
    
    # Training data prediction and evaluation
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training Data Accuracy:", train_accuracy * 100)
    
    # Testing data prediction and evaluation
    y_test_pred = classifier.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("\nTesting Data Accuracy:", test_accuracy * 100)'''













































#used XGB beacuse of better accuracy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMLData.csv", index_col=None)

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Scale the specified columns in the original DataFrame
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=50)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=50)
xgb_classifier.fit(X_train, y_train)

# Training data prediction and evaluation
y_train_pred = xgb_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Data Accuracy:", train_accuracy * 100)

# Testing data prediction and evaluation
y_test_pred = xgb_classifier.predict(X_test)

# Classification Report
print("\nClassification Report:")
classification_rep = classification_report(y_test, y_test_pred)
print(classification_rep)

# Extracting F1 score from classification report
f1 = f1_score(y_test, y_test_pred)
print("F1 Score:", f1)

test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTesting Data Accuracy:", test_accuracy * 100)

# Feature Importance
feature_importance = xgb_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=X_train.columns[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Correlation Matrix
correlation_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# AUC-ROC Curve
y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()





# predicting loan status
'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMLData.csv", index_col=None)

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the selected features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=50)
xgb_classifier.fit(X_resampled, y_resampled)

# Function to preprocess user input
def preprocess_input(user_input):
    # Scale the specified columns
    user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df = user_input_df.copy()
    scaled_user_input_df[features_to_scale] = scaler.transform(scaled_user_input_df[features_to_scale])
    return user_input_df, scaled_user_input_df

# Function to predict loan status and its probability based on user input
def predict_loan_status(user_input):
    # Preprocess user input
    _, scaled_input_df = preprocess_input(user_input)

    # Predict loan status probabilities
    loan_status_probs = xgb_classifier.predict_proba(scaled_input_df)[0]

    # Get the probability of approval
    probability_of_approval = loan_status_probs[1]

    # Convert probability to percentage
    probability_percentage = round(probability_of_approval * 100, 2)

    # Predict loan status
    loan_status = "Approved" if probability_of_approval >= 0.5 else "Rejected"

    return loan_status, probability_percentage

# Get input from the user
print("Please provide the following details:")
no_of_dependents = int(input("Number of dependents: "))
education = int(input("Education level (0 for Not Graduate, 1 for Graduate): "))
self_employed = int(input("Self-employed (0 for No, 1 for Yes): "))
income_annum = float(input("Annual income: "))
loan_amount = float(input("Loan amount: "))
loan_term = int(input("Loan term (in months): "))
cibil_score = float(input("CIBIL score: "))
residential_assets_value = float(input("Value of residential assets: "))
commercial_assets_value = float(input("Value of commercial assets: "))
luxury_assets_value = float(input("Value of luxury assets: "))
bank_asset_value = float(input("Value of assets in the bank: "))

# Prepare user input as a dictionary
user_input = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# Predict loan status and its probability
predicted_status, probability_percentage = predict_loan_status(user_input)
print("\nPredicted Loan Status:", predicted_status)
print("Probability of Loan Approval:", probability_percentage, "%")'''







#ccccccccccccccccccccc
'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['LoanML']  # Change 'your_database_name' to your actual database name
collection = db['LoanApproval']  # Change 'your_collection_name' to your actual collection name

# Load data from MongoDB
cursor = collection.find({}, {'_id': 0})
data = pd.DataFrame(list(cursor))

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the selected features
data[features_to_scale[:-1]] = scaler.fit_transform(data[features_to_scale[:-1]])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=50)
xgb_classifier.fit(X_resampled, y_resampled)

# Function to preprocess user input
def preprocess_input(user_input):
    # Scale the specified columns
    user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df = user_input_df.copy()
    scaled_user_input_df[features_to_scale[:-1]] = scaler.transform(scaled_user_input_df[features_to_scale[:-1]])
    return user_input_df, scaled_user_input_df

# Function to predict loan status and its probability based on user input
def predict_loan_status(user_input):
    # Preprocess user input
    _, scaled_input_df = preprocess_input(user_input)

    # Predict loan status probabilities
    loan_status_probs = xgb_classifier.predict_proba(scaled_input_df)[0]

    # Get the probability of approval
    probability_of_approval = loan_status_probs[1]

    # Convert probability to percentage
    probability_percentage = round(probability_of_approval * 100, 2)

    # Predict loan status
    loan_status = "Approved" if probability_of_approval >= 0.5 else "Rejected"

    return loan_status, probability_percentage

# Get input from the user
print("Please provide the following details:")
no_of_dependents = int(input("Number of dependents: "))
education = int(input("Education level (0 for Not Graduate, 1 for Graduate): "))
self_employed = int(input("Self-employed (0 for No, 1 for Yes): "))
income_annum = float(input("Annual income: "))
loan_amount = float(input("Loan amount: "))
loan_term = int(input("Loan term (in months): "))
cibil_score = float(input("CIBIL score: "))
residential_assets_value = float(input("Value of residential assets: "))
commercial_assets_value = float(input("Value of commercial assets: "))
luxury_assets_value = float(input("Value of luxury assets: "))
bank_asset_value = float(input("Value of assets in the bank: "))

# Prepare user input as a dictionary
user_input = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# Predict loan status and its probability
predicted_status, probability_percentage = predict_loan_status(user_input)
print("\nPredicted Loan Status:", predicted_status)
print("Probability of Loan Approval:", probability_percentage, "%")'''


#PREDICTION CODE OF LOAN APPROVAL WITH API
'''from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['LoanML']  # Change 'your_database_name' to your actual database name
collection = db['LoanApproval']  # Change 'your_collection_name' to your actual collection name

# Load data from MongoDB
cursor = collection.find({}, {'_id': 0})
data = pd.DataFrame(list(cursor))

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the selected features
data[features_to_scale[:-1]] = scaler.fit_transform(data[features_to_scale[:-1]])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=50)
xgb_classifier.fit(X_resampled, y_resampled)

# Function to preprocess user input
def preprocess_input(user_input):
    # Scale the specified columns
    scaled_user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df[features_to_scale[:-1]] = scaler.transform(scaled_user_input_df[features_to_scale[:-1]])
    return scaled_user_input_df

# Function to predict loan status and its probability based on user input
def predict_loan_status(user_input):
    # Preprocess user input
    scaled_input_df = preprocess_input(user_input)

    # Predict loan status probabilities
    loan_status_probs = xgb_classifier.predict_proba(scaled_input_df)[0]

    # Get the probability of approval
    probability_of_approval = loan_status_probs[1]

    # Convert probability to percentage
    probability_percentage = round(probability_of_approval * 100, 2)

    # Predict loan status
    loan_status = "Approved" if probability_of_approval >= 0.5 else "Rejected"

    return loan_status, probability_percentage

# Create Flask app
app = Flask(__name__)

# Define route for loan prediction
@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    user_input = request.get_json()  # Get user input from JSON request

    # Predict loan status and probability
    predicted_status, probability_percentage = predict_loan_status(user_input)

    # Create response JSON
    response = {
        "loan_status": predicted_status,
        "probability_of_approval": probability_percentage
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)'''




'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient

# Establish a connection to MongoDB
client = MongoClient('localhost', 27017)
db = client['LoanML']
collection = db['LoanApproval']

# Retrieve data from MongoDB Compass, excluding the _id field
cursor = collection.find({}, {'_id': 0})
data = pd.DataFrame(list(cursor))

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the selected features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=50)
xgb_classifier.fit(X_resampled, y_resampled)

# Function to preprocess user input
def preprocess_input(user_input):
    # Scale the specified columns
    scaled_user_input_df = user_input.copy()
    scaled_user_input_df[features_to_scale] = scaler.transform([list(user_input.values())])
    return scaled_user_input_df

# Function to predict loan status and its probability based on user input
def predict_loan_status(user_input):
    # Preprocess user input
    scaled_input_df = preprocess_input(user_input)

    # Predict loan status probabilities
    loan_status_probs = xgb_classifier.predict_proba(scaled_input_df)[0]

    # Get the probability of approval
    probability_of_approval = loan_status_probs[1]

    # Convert probability to percentage
    probability_percentage = round(probability_of_approval * 100, 2)

    # Predict loan status
    loan_status = "Approved" if probability_of_approval >= 0.5 else "Rejected"

    return loan_status, probability_percentage

# Get input from the user
print("Please provide the following details:")
no_of_dependents = int(input("Number of dependents: "))
education = int(input("Education level (0 for Not Graduate, 1 for Graduate): "))
self_employed = int(input("Self-employed (0 for No, 1 for Yes): "))
income_annum = float(input("Annual income: "))
loan_amount = float(input("Loan amount: "))
loan_term = int(input("Loan term (in months): "))
cibil_score = float(input("CIBIL score: "))
residential_assets_value = float(input("Value of residential assets: "))
commercial_assets_value = float(input("Value of commercial assets: "))
luxury_assets_value = float(input("Value of luxury assets: "))
bank_asset_value = float(input("Value of assets in the bank: "))

# Prepare user input as a dictionary
user_input = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# Predict loan status and its probability
predicted_status, probability_percentage = predict_loan_status(user_input)
print("\nPredicted Loan Status:", predicted_status)
print("Probability of Loan Approval:", probability_percentage, "%")'''




# predicting loan status and checking the scaling

'''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the data without setting the index column
data = pd.read_csv("LoanMLData.csv", index_col=None)

# Select features for scaling
features_to_scale = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value']

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the selected features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Split the data into features (X) and the target variable (y)
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_resampled, y_resampled)

# Function to preprocess user input
def preprocess_input(user_input):
    # Scale the specified columns
    user_input_df = pd.DataFrame([user_input])
    scaled_user_input_df = user_input_df.copy()
    scaled_user_input_df[features_to_scale] = scaler.transform(scaled_user_input_df[features_to_scale])
    return user_input_df, scaled_user_input_df

# Function to predict loan status based on user input
def predict_loan_status(user_input):
    # Preprocess user input
    _, scaled_input_df = preprocess_input(user_input)

    # Predict loan status
    loan_status = xgb_classifier.predict(scaled_input_df)[0]

    if loan_status == 1:
        return "Approved"
    else:
        return "Rejected"

# Get input from the user
print("Please provide the following details:")
no_of_dependents = int(input("Number of dependents: "))
education = int(input("Education level (0 for Not Graduate, 1 for Graduate): "))
self_employed = int(input("Self-employed (0 for No, 1 for Yes): "))
income_annum = float(input("Annual income: "))
loan_amount = float(input("Loan amount: "))
loan_term = int(input("Loan term (in months): "))
cibil_score = float(input("CIBIL score: "))
residential_assets_value = float(input("Value of residential assets: "))
commercial_assets_value = float(input("Value of commercial assets: "))
luxury_assets_value = float(input("Value of luxury assets: "))
bank_asset_value = float(input("Value of assets in the bank: "))

# Prepare user input as a dictionary
user_input = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# Print user input before scaling
print("\nUser Input Data (Before Scaling):")
print(pd.DataFrame([user_input]))

# Predict loan status
predicted_status = predict_loan_status(user_input)
print("\nPredicted Loan Status:", predicted_status)

# Print user input after scaling
_, scaled_input_df = preprocess_input(user_input)
print("\nUser Input Data (After Scaling):")
print(scaled_input_df)'''















































































































































































































































