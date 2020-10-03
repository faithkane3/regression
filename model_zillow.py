import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
from math import sqrt

import features_zillow


def modeling_function(X_train, y_train, X_test, y_test):
    predictions=pd.DataFrame({"actual":y_train.home_value}).reset_index(drop=True)
    predictions_test=pd.DataFrame({"actual":y_test.home_value}).reset_index(drop=True)

    #model 1 - 3 features
    lm1=LinearRegression()
    lm1.fit(X_train,y_train)
    lm1_predictions=lm1.predict(X_train)
    predictions["lm1"]=lm1_predictions

    #model test
    lm1_test=LinearRegression()
    lm1_test.fit(X_test,y_test)
    lm1_test_predictions=lm1_test.predict(X_test)
    predictions_test["lm1_test"]=lm1_test_predictions

    #model baseline
    predictions["lm_baseline"] = y_train.home_value.mean()
    predictions_test["lm_baseline_test"] = y_test.home_value.mean()

    #error_delta columns

    return predictions, predictions_test


def plot_residuals(X, y):
    '''
    Plots the residuals of a model that uses X to predict y. Note that we don't
    need to make any predictions ourselves here, seaborn will create the model
    and predictions for us under the hood with the `residplot` function.
    '''
    return sns.residplot(X, y)

