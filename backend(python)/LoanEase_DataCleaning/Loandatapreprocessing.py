import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("LoanEase_DataCleaning//loan_approval_dataset.csv")
#print(data)
df=pd.DataFrame(data)
#print(df)

#---------Data Cleaning----------------#
df.columns = df.columns.str.replace(' ', '')

#checking null values 
#print(df.isnull().sum()) #no null values found

#checking duplicate values in a dataset
#print(df[df.duplicated(subset=['loan_id'])]) #no duplicate values

#dropping column loan_id [not useful]
df.drop(columns=['loan_id'], inplace=True)

#checking datatypes of columns 
#print(df.dtypes)

# check if some integer columns contains some negative values or not.
# Filter integer columns
integer_columns = df.select_dtypes(include='int64')

# Check if any integer column contains negative values
negative_values_exist = (integer_columns < 0).any()
#print(negative_values_exist)

#negative values exit in residential asset values column, let's remove it
df= df[df['residential_assets_value'] > 0]
print(df)

df.to_csv('LoanEase_DataCleaning//loandata.csv', index=False) 
