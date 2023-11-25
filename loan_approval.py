import numpy as np
import pandas as pd
from sklearn import linear_model

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/SP II/loan_approval_dataset.csv')

df


likhon=linear_model.LinearRegression()

likhon.fit(df[['No_of_dependents','Education','Self_employed','Income_annual','Loan_amount','Loan_tearm','Cibil_score','Residential_assets','Commercial_assets','Luxurious_assets','Bank_assets']],df.Loan_status)

likhon.predict([[3,1,0,500000,1000000,36,650,0,0,400000,430000]])

likhon.coef_

likhon.intercept_

-3.09028668e-01*3+7.50932145e-01-3.82921709e-06*500000+1.01585388e-06*1000000-1.00972803e+00*36+2.16739624e-01*650+1.78948628e-07*400000+ 2.25247708e-07*430000-56.98328506880289
