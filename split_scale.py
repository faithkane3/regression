import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


def split_my_data(df):
    '''
    This function performs a 3-way split returning my train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


def standard_scaler(columns_to_scale, train, validate, test):
    """
    Takes in train, validate, and test dfs with numeric values only
    Returns scaler, train_scaled, validate_scaled, test_scaled dfs
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    scaler = StandardScaler().fit(train[columns_to_scale])
    
    train_scaled = pd.concat([
                        train,
                        pd.DataFrame(scaler.transform(train[columns_to_scale]), 
                        columns=new_column_names, 
                        index=train.index)],
                        axis=1)
    
    validate_scaled = pd.concat([
                        validate,
                        pd.DataFrame(scaler.transform(validate[columns_to_scale]), 
                        columns=new_column_names, 
                        index=validate.index)],
                        axis=1)
    
    test_scaled = pd.concat([
                        test,
                        pd.DataFrame(scaler.transform(test[columns_to_scale]), 
                        columns=new_column_names, 
                        index=test.index)],
                        axis=1)
    
    return scaler, train_scaled, validate_scaled, test_scaled



def min_max_scaler(X_train, X_test):
    """Transforms features by scaling each feature to a given range.
       Takes in X_train and X_test,
       Returns the scaler and X_train_scaled and X_test_scaled within range.
       Sensitive to outliers.
    """
    scaler = (MinMaxScaler(copy=True, 
                           feature_range=(0,1))
                          .fit(X_train))
    X_train_scaled = (pd.DataFrame(scaler.transform(X_train), 
                      columns=X_train.columns, 
                      index=X_train.index))
    X_test_scaled = (pd.DataFrame(scaler.transform(X_test), 
                     columns=X_test.columns,
                     index=X_test.index))
    return scaler, X_train_scaled, X_test_scaled



def iqr_robust_scaler(X_train, X_test):
    """Scales features using stats that are robust to outliers
       by removing the median and scaling data to the IQR.
       Takes in a X_train and X_test,
       Returns the scaler and X_train_scaled and X_test_scaled.
    """
    scaler = (RobustScaler(quantile_range=(25.0,75.0), 
                           copy=True, 
                           with_centering=True, 
                           with_scaling=True)
                          .fit(X_train))
    X_train_scaled = (pd.DataFrame(scaler.transform(X_train), 
                      columns=X_train.columns, 
                      index=X_train.index))
    X_test_scaled = (pd.DataFrame(scaler.transform(X_test), 
                     columns=X_test.columns,
                     index=X_test.index))
    return scaler, X_train_scaled, X_test_scaled


def uniform_scaler(X_train, X_test):
    """Quantile transformer, non_linear transformation - uniform.
       Reduces the impact of outliers, smooths out unusual distributions.
       Takes in a X_train, X_validate, and X_test dfs
       Returns the scaler, X_train_scaled, X_validate_scaled, X_test_scaled
    """
    scaler = (QuantileTransformer(n_quantiles=100, 
                                  output_distribution='uniform', 
                                  random_state=123, copy=True)
                                  .fit(X_train))
    
    X_train_scaled = (pd.DataFrame(scaler.transform(X_train), 
                      columns=X_train.columns, 
                      index=X_train.index))
    
    X_validate_scaled = (pd.DataFrame(scaler.transform(X_validate), 
                      columns=X_validate.columns, 
                      index=X_validate.index))
    
    X_test_scaled = (pd.DataFrame(scaler.transform(X_test), 
                     columns=X_test.columns,
                     index=X_test.index))
    
    return scaler, X_train_scaled, X_test_scaled


def gaussian_scaler(X_train, X_test):
    """Transforms and then normalizes data.
       Takes in X_train and X_test dfs, 
       yeo_johnson allows for negative data,
       box_cox allows positive data only.
       Returns Zero_mean, unit variance normalized X_train_scaled and X_test_scaled and scaler.
    """
    scaler = (PowerTransformer(method='yeo-johnson', 
                               standardize=False, 
                               copy=True)
                              .fit(X_train))
    X_train_scaled = (pd.DataFrame(scaler.transform(X_train), 
                      columns=X_train.columns, 
                      index=X_train.index))
    X_test_scaled = (pd.DataFrame(scaler.transform(X_test), 
                     columns=X_test.columns,
                     index=X_test.index))
    return scaler, X_train_scaled, X_test_scaled