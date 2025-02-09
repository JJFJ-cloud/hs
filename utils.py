# utils.py

import pandas as pd
from sklearn import datasets
from imblearn.over_sampling import SMOTE

def get_dataset( new_data=None):
        if new_data is None:
            raise ValueError("For 'Custom Dataset', 'new_data' must be provided.")
        data = new_data
        # data = data.replace(-9999, 0)
        feature_df = data.drop(columns=['Level'])  # 假设 'cuisine' 是目标变量
        labels_df = data['Level']
        oversample = SMOTE(sampling_strategy='auto', k_neighbors=5)
        transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
        data = pd.concat([transformed_feature_df, transformed_label_df], axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].values
        return X, y, data

        