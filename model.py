#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:06:09 2020

@author: Ryan Wong
"""

import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('appstore_games_cleaned.csv')

# Split into explanatory variables and response variable
x = df.drop('Average User Rating', axis =1)
y = df['Average User Rating'].values

# Split data: 50% training set, 50% testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# Train model
regressor = LinearRegression()  
regressor.fit(x_train, y_train)

# Prediction on test data
y_pred = regressor.predict(x_test)

# Regression Coefficients
coefficients = regressor.coef_

# Difference between the actual value and predicted value
diff = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Calculate R-squared
diff['Residuals'] = diff['Actual'] - diff['Predicted']
avg_actual = diff['Actual'].mean()
diff['Diff_Actual_Avg_Squared'] = (diff['Actual'] - avg_actual) ** 2
SST = diff['Diff_Actual_Avg_Squared'].sum()

diff['Squared_Diff'] = diff['Residuals'] ** 2
SSE = diff['Squared_Diff'].sum()

r_squared = 1 - (SSE/SST) # 0.1678

# Mean Absolute Error
MAE = mean_absolute_error(diff['Actual'], diff['Predicted'])
print(MAE) # 0.3706

# Backward Elimination
import numpy as np
x_train = np.append (arr=np.ones([x_train.shape[0],1]).astype(int), values = x_train, axis = 1)

import statsmodels.api as sm
# significance level = 0.05
X_opt = [0,2,3,9,10,13,14,15,17,18,31,40,46,51]
regressor = sm.OLS(endog = y_train, exog = x_train[:,X_opt]).fit()
print(regressor.summary())



# Model incorporating only the attributes obtained from backwards elimination

# Split into explanatory variables and response variable
x_opt = df[['subtitle_yes_no','In-App-Q1','< 12','< 17','size_Q2','size_Q3','size_Q4',
            'Adventure','Board','Magazines & Newspapers','Reference','days_since_update']].values
y_opt = df['Average User Rating'].values

# Split data: 50% training set, 50% testing set
x_opt_train, x_opt_test, y_opt_train, y_opt_test = train_test_split(x_opt, y_opt, test_size=0.5, random_state=0)

# Train model
regressor2 = LinearRegression()  
regressor2.fit(x_opt_train, y_opt_train)

# Prediction on test data
y_opt_pred = regressor2.predict(x_opt_test)

# Regression Coefficients
coefficients = regressor2.coef_

# Difference between the actual value and predicted value
diff = pd.DataFrame({'Actual': y_opt_test, 'Predicted': y_opt_pred})

# Calculate R-squared
from sklearn.metrics import r2_score
print(r2_score(y_opt_test, y_opt_pred)) # 0.1684

# Mean Absolute Error
MAE = mean_absolute_error(y_opt_test, y_opt_pred)
print(MAE) # 0.3675

