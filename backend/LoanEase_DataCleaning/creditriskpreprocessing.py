import numpy as np
import pandas as pd

data=pd.read_csv("LoanEase_DataCleaning//credit_risk_dataset.csv")
df=pd.DataFrame(data)
#print(df)
#print("\nINFO OF CREDIT RISK DATAFRAME")
#print(df.info())

#change income dtype from int to float
df['person_income'] = df['person_income'].astype(float)


#Finding duplicate rows in a datset (found 165 rows duplicate)
dups = df.duplicated()
#print(df[dups])

#deleting duplicate rows from a datset
df.drop_duplicates(inplace=True)
#print(df)

#finding missing values, found 887 in emp_length & 3095 in int_rate
#print(df.isnull().sum())


#filling missing value using mean of corresponding column
df = df.fillna({'person_emp_length':df['person_emp_length'].mean(), 'loan_int_rate':df['loan_int_rate'].mean()})  

df['person_emp_length'] = df['person_emp_length'].round().astype(int)
#description of dataset#
#print("\nDESCRIPTION OF CREDIT RISK DATAFRAME")
#print(df.describe())

# (checking if the numeric columns contains any negative values or not)

# Get a list of numeric columns of types float64 and int64
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Initialize a flag to indicate if any negative value is found
negative_found = False

# Iterate through numeric columns
for column in numeric_columns:
    # Check if any negative value exists in the column
    if (df[column] < 0).any():
        negative_found = True
        print(f"Negative values found in column '{column}'")

# Check if any negative value was found in any column
if not negative_found:
    print("No negative values found in any numeric column.")

#[ result = no negative values exist any any numeric column]    
 
# (Fitting Age Range and Employbility Range)
# Filter rows based on age range
df = df[(df['person_age'] >= 18) & (df['person_age'] <= 70)]

# Filter rows based on employment length range
df = df[(df['person_emp_length'] >= 0) & (df['person_emp_length'] <= 50)]

#df.to_csv('LoanEase_DataCleaning//creditriskdata.csv', index=False) 
#print(df.info())

