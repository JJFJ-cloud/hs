# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def replace_outliers_with_mean(data, outlier_value=-9999):
    for column in data.columns:
        column_mean = data[data[column] != outlier_value][column].mean()
        data[column] = data[column].replace(outlier_value, column_mean)
    return data

import pandas as pd
import numpy as np

def clean_outliers(df, handle_strategy):
    df_cleaned = df.copy()
    
    for column in df.select_dtypes(include=[np.number]).columns:  # 只处理数值型列
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        if outliers.any():  # 检查是否有异常值
            df_non_outliers = df[~outliers][column]
            
            if handle_strategy == 'median':
                median = df_non_outliers.median()
                df_cleaned.loc[outliers, column] = median
            elif handle_strategy == 'mean':
                mean = df_non_outliers.mean()
                df_cleaned.loc[outliers, column] = mean
            elif handle_strategy == 'mode':
                mode_value = df_non_outliers.mode().iloc[0] if not df_non_outliers.mode().empty else df_non_outliers.iloc[0]
                df_cleaned.loc[outliers, column] = mode_value
            else:
                raise ValueError("Invalid handle_strategy. Choose from 'median', 'mean', or 'mode'.")
    
    return df_cleaned

def get_scaler(scaler_name):
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }
    return scalers.get(scaler_name)

def standardize_data(data, scaler, features_to_scale):
    X_scaled = data.copy()
    scaled_values = scaler.fit_transform(X_scaled[features_to_scale])
    X_scaled = pd.DataFrame(scaled_values, columns=features_to_scale)
    return X_scaled
def standardize_datayc(data, scaler):
    # 只处理数值型列
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    X_scaled = data[numeric_columns].copy()
    
    # 标准化
    scaled_values = scaler.fit_transform(X_scaled)
    X_scaled = pd.DataFrame(scaled_values, index=data.index, columns=numeric_columns)
    
    # 将非数值型列合并回数据框
    for column in data.columns:
        if column not in numeric_columns:
            X_scaled[column] = data[column]
    
    return X_scaled