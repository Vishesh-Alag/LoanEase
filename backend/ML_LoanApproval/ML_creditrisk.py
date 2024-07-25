# Ready the Data For ML Training and Testing
# preprocessing
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data=pd.read_csv("LoanEase_DataCleaning/creditriskdata.csv")
df=pd.DataFrame(data)


# Perform label encoding
label_mapping = {
    "person_home_ownership": {'OWN': 0, 'MORTGAGE': 1, 'RENT': 2, 'OTHER': 3},
    "loan_intent": {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5},
    "cb_person_default_on_file": {'N': 0, 'Y': 1}
}
df.replace(label_mapping, inplace=True)

# Plot box plots for numerical features
numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
plt.figure(figsize=(10, 8))
sns.boxplot(data=data[numerical_features])
plt.title('Box plot for Numerical Features')
plt.xticks(rotation=45)
plt.show()

# Calculate z-scores for numerical columns
z_scores = stats.zscore(df[numerical_features])

# Define threshold for outliers
threshold = 3

# Find indices of outliers
outlier_indices = np.where(np.abs(z_scores) > threshold)

# Get unique row indices containing outliers
outlier_row_indices = np.unique(outlier_indices[0])

# Remove rows with outliers
df = df.drop(outlier_row_indices)

# Print number of outliers removed
num_outliers_removed = len(outlier_row_indices)
print("Number of rows with outliers removed:", num_outliers_removed)'''

#df.to_csv('CreditRiskMLData.csv',index=False)


#check the majority and miniority class for target variable
'''import pandas as pd
import numpy as np
data=pd.read_csv("LoanEase_DataCleaning/creditriskdata.csv")
df=pd.DataFrame(data)
defaulted_loans = df.loc[df['loan_status']==1]
non_defaulted_loans = df.loc[df['loan_status']==0]


print( defaulted_loans.shape[0])
print(non_defaulted_loans.shape[0])'''


#comparison between different models
#XGBoost performs best with accuracy 92.72% & f-1 score of 0.9250
'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score,f1_score

# Load the dataset
data = pd.read_csv("CreditRiskMLData.csv")

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = data[columns_to_scale]

# Scale the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
data[columns_to_scale] = X_scaled

# Split the data into features and target variable
X = data.drop(columns=["loan_status"])
y = data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=50)

# Initialize and train the models with fewer iterations or simpler configurations
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),  
    "Gradient Boosting": GradientBoostingClassifier(),  
    "XGBoost": XGBClassifier()  
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)  # Calculate F1 score
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------------")'''


#choosing the best one XGB and its feature importance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("CreditRiskMLData.csv")

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = data[columns_to_scale]

# Scale the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
data[columns_to_scale] = X_scaled

# Split the data into features and target variable
X = data.drop(columns=["loan_status"])
y = data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=50)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Plot feature importance
feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Most Important Features - XGBoost')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Testing data prediction and evaluation
y_pred = xgb_model.predict(X_test)

# Classification Report
print("\nClassification Report:")
classification_rep = classification_report(y_test, y_pred)
print(classification_rep)

# Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTesting Data Accuracy:", test_accuracy * 100)

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt="d", xticklabels=["Not Default", "Default"], yticklabels=["Not Default", "Default"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Correlation Matrix Visualization
plt.figure(figsize=(10, 8))
correlation_matrix = X_train.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# AUC-ROC Curve
y_pred_proba = xgb_model.predict_proba(X_test)[:,1] # Probability estimates of the positive class
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



# Rechecking the Values whether the training and testing (user input data) are scaled or not before making prediction
'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset for training
training_data = pd.read_csv("CreditRiskMLData.csv")

# Print training data before scaling and training
print("\nTraining data before scaling and training:")
print(training_data.head(1))

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns for training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

print("\nTraining data after scaling:")
print(training_data.head(1))

print("\nIs training data scaled? ", scaler.scale_ is not None)

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model with random_state=50
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

print("\nTraining completed.")

# Take input data from user (testing data)
print("\nEnter the testing data:")
input_data = {}
for column in X_train.columns:
    input_data[column] = [float(input(f"Enter {column}: "))]

# Create DataFrame for testing data
testing_data = pd.DataFrame(input_data)

# Scale testing data using the same scaler fitted on training data
X_test_scaled = scaler.transform(testing_data[columns_to_scale])
testing_data[columns_to_scale] = X_test_scaled

print("\nIs testing data scaled? ", scaler.scale_ is not None)
print("\n",testing_data)

# Predict using the trained model
predictions = xgb_model.predict(testing_data)

print("\nPredictions:", predictions)'''




#prediction on user input data

'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset for training
training_data = pd.read_csv("CreditRiskMLData.csv")

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

# Take input data from user (testing data)
print("Enter the testing data:")
input_data = {}
for column in X_train.columns:
    input_data[column] = [float(input(f"Enter {column}: "))]

# Create DataFrame for testing data
testing_data = pd.DataFrame(input_data)

# Scale testing data using the same scaler fitted on training data
X_test_scaled = scaler.transform(testing_data[columns_to_scale])
testing_data[columns_to_scale] = X_test_scaled

# Predict using the trained model
predictions = xgb_model.predict(testing_data)

print("Predictions:", predictions)'''


# refine version of prediction on user input data
#includes percentage chances also
'''import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset for training
training_data = pd.read_csv("CreditRiskMLData.csv")
#print(training_data.info())

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns for training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model with random_state=50
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

# Take input data from user (testing data)
print("\nEnter the testing data:")
input_data = {}
for column in X_train.columns:
    input_data[column] = [float(input(f"Enter {column}: "))]

# Create DataFrame for testing data
testing_data = pd.DataFrame(input_data)

# Scale testing data using the same scaler fitted on training data
X_test_scaled = scaler.transform(testing_data[columns_to_scale])
testing_data[columns_to_scale] = X_test_scaled

# Predict using the trained model
predictions = xgb_model.predict(testing_data)
probabilities = xgb_model.predict_proba(testing_data)

# Print predictions and probabilities
if predictions[0] == 1:
    print("\nPerson is Likely to Default.")
else:
    print("\nPerson is Not Likely to Default.")

print("Probability of default: {:.2f}%".format(probabilities[0][1] * 100))'''






# refine version of prediction on user input data
#includes percentage chances also WITH API
'''from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load the dataset for training
training_data = pd.read_csv("CreditRiskMLData.csv")

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns for training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model with random_state=50
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

@app.route('/predict_creditrisk', methods=['POST'])
def predict():
    try:
        # Take input data from user (testing data)
        input_data = request.json
        
        # Create DataFrame for testing data
        testing_data = pd.DataFrame([input_data])
        
        # Scale testing data using the same scaler fitted on training data
        X_test_scaled = scaler.transform(testing_data[columns_to_scale])
        testing_data[columns_to_scale] = X_test_scaled
        
        # Predict using the trained model
        predictions = xgb_model.predict(testing_data)
        probabilities = xgb_model.predict_proba(testing_data)
        
        # Prepare response
        if predictions[0] == 1:
            result = {"prediction": "Person is Likely to Default", "probability_of_default": round(probabilities[0][1] * 100, 2)}
        else:
            result = {"prediction": "Person is Not Likely to Default", "probability_of_default": round(probabilities[0][1] * 100, 2)}
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)'''



# refine version of prediction on user input data
#includes percentage chances also WITH API WITH MONGODB
''''from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['LoanML']
collection = db['CreditRisk']

# Retrieve data from MongoDB collection and remove the _id field
cursor = collection.find({}, {'_id': 0})
data = list(cursor)
training_data = pd.DataFrame(data)

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns for training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model with random_state=50
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

@app.route('/predict_creditrisk', methods=['POST'])
def predict():
    try:
        # Take input data from user (testing data)
        input_data = request.json
        
        # Create DataFrame for testing data
        testing_data = pd.DataFrame([input_data])
        
        # Scale testing data using the same scaler fitted on training data
        X_test_scaled = scaler.transform(testing_data[columns_to_scale])
        testing_data[columns_to_scale] = X_test_scaled
        
        # Predict using the trained model
        predictions = xgb_model.predict(testing_data)
        probabilities = xgb_model.predict_proba(testing_data)
        
        # Prepare response
        if predictions[0] == 1:
            result = {"prediction": "Person is Likely to Default", "probability_of_default": round(probabilities[0][1] * 100, 2)}
        else:
            result = {"prediction": "Person is Not Likely to Default", "probability_of_default": round(probabilities[0][1] * 100, 2)}
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)'''





# refine version of prediction on user input data
#includes percentage chances also WITH API WITH MONGODB
#with ranges (high,low,medium)
'''from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['LoanML']
collection = db['CreditRisk']

# Retrieve data from MongoDB collection and remove the _id field
cursor = collection.find({}, {'_id': 0})
data = list(cursor)
training_data = pd.DataFrame(data)

# Remove the columns not needed for scaling
columns_to_scale = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
X_scaling = training_data[columns_to_scale]

# Scale the selected columns for training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaling)
training_data[columns_to_scale] = X_scaled

# Split the training data into features and target variable
X_train = training_data.drop(columns=["loan_status"])
y_train = training_data["loan_status"]

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=50)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the XGBoost model with random_state=50
xgb_model = XGBClassifier(random_state=50)
xgb_model.fit(X_resampled, y_resampled)

@app.route('/predict_creditrisk', methods=['POST'])
def predict():
    try:
        # Take input data from user (testing data)
        input_data = request.json
        
        # Create DataFrame for testing data
        testing_data = pd.DataFrame([input_data])
        
        # Scale testing data using the same scaler fitted on training data
        X_test_scaled = scaler.transform(testing_data[columns_to_scale])
        testing_data[columns_to_scale] = X_test_scaled
        
        # Predict using the trained model
        predictions = xgb_model.predict(testing_data)
        probabilities = xgb_model.predict_proba(testing_data)
        
        # Prepare response
        if predictions[0] == 1:
            if probabilities[0][1] > 0.7:
                result = {"prediction": "High Risk To DEFAULT", "probability_of_default": round(probabilities[0][1] * 100, 2)}
            elif probabilities[0][1] > 0.4:
                result = {"prediction": "Medium Risk To DEFAULT", "probability_of_default": round(probabilities[0][1] * 100, 2)}
            else:
                result = {"prediction": "Low Risk To DEFAULT", "probability_of_default": round(probabilities[0][1] * 100, 2)}
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)'''










