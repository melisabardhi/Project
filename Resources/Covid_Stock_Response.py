# -*- coding: utf-8 -*-

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn import metrics
from scipy.stats import pearsonr

"""
You will need to uncomment the corresponding data for the data you are inputing
A checklist is provided and the neccessary lines are marked with "###" within the comment

CHECKLIST:
    1) In "DATASET" section uncomment the line of code you will be using to read the csv (i.e. dataset= ...)
    2) In "Variables" section uncomment the corresponding X variable line of code that matches with the DATASET (i.e. X= ...)
    3) In "Coefficients of Attributes" section uncomment the line that matches the "Variables" section "X="
    4) In the "Print Out Errors and Stat info" sectionuncomment the line that matches the "Variables" section "X="
"""



#=============== DATASET ==============================================================================================================
###     All companies Close
dataset = pd.read_csv('Your file path')
###     Without Netflix
#dataset = pd.read_csv('your file path')

#D:\Github\Covid_Stock_Response
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')
description=dataset.describe()
#======================================================================================================================================


#=============== "Variables" ============================================================================================================

###     All companies Close
# With Netflix
X=dataset[['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','NFLX Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close']].values

###     Without Netflix
#X=dataset[['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close']].values


y=dataset['SP Close'].values
#======================================================================================================================================


#=============== Regression Model =====================================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#======================================================================================================================================


#=============== Coefficients of Attributes ===========================================================================================
###     All companies Close
coeff_df = pd.DataFrame(regressor.coef_, ['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','NFLX Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close'] , columns=['Coefficient'])  

###     Without Netflix
#coeff_df = pd.DataFrame(regressor.coef_, ['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close'] , columns=['Coefficient'])  
#======================================================================================================================================


#=============== T-stat ===============================================================================================================
tstat=[]
ymean= description.iloc[1,0]
sy= description.iloc[2,0]
for n in range(1,len(coeff_df)+1):
    xmean=description.iloc[1,n]
    sx= description.iloc[2,n]
    t= (ymean-xmean)/ np.sqrt(((sx**2)/317) + ((sy**2)/317))
    tstat.append(t)
coeff_df['T-Stat']= tstat   
#======================================================================================================================================

    
#=============== Correlation ==========================================================================================================
correlation=[]
for i in range(len(coeff_df)):    
    corr, _ = pearsonr(X[:,i], y)
    correlation.append(corr)

coeff_df['Correlation']= correlation
print(coeff_df)
print('Intercept: ',regressor.intercept_)
#======================================================================================================================================



#=============== Prediction of Test Data ==============================================================================================
y_pred = regressor.predict(X_test)
#======================================================================================================================================


#=============== Difference Between Actual and Predicted ==============================================================================
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(64)
#======================================================================================================================================


#=============== Comparison Between Actual and Predicted ==============================================================================
df1.plot(kind='line',figsize=(15,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#======================================================================================================================================


#=============== Print Out Errors and Stat info =====================================================================================
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('r2:', metrics.r2_score(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
[fstat, pvalue] = f_regression(X, y) 
###     All companies
FandPdf = pd.DataFrame(index=['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','NFLX Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close'])  
###     Without Netflix
#FandPdf = pd.DataFrame(index=['DAL Close','AAL Close','BLU Close','CCL Close','NCLH Close','RCL Close', 'MAR Close','HLT Close','PK Close','XOM Close','CVX Close','BP Close','DIS Close','CNK Close','PG Close', 'JNJ Close', 'NSRGY Close','ZM Close', 'MSFT Close','LOGM Close','FB Close','TWTR Close','SNAP Close'])
FandPdf['F-Stat'] = fstat
FandPdf['P-value'] = pvalue
print(FandPdf)
#======================================================================================================================================