# model_training.py

import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def add_parameter_ui(clf_name,prefix):
    params = dict()
    if clf_name == 'SVM':
        params['C'] = st.slider('正则化参数C', 0.01, 10.0, value=1.0, key=f'{prefix}_svm_c')  # 默认值1.0
        params['kernel'] = st.selectbox('核函数', ['rbf', 'linear'], key=f'{prefix}_svm_kernel')  # 默认选项为'rbf'和'linear'
        params['gamma'] = st.selectbox('gamma参数', ['scale', 'auto'], key=f'{prefix}_svm_gamma')  # 默认选项为'scale'和'auto'
    elif clf_name == 'KNN':
        params['n_neighbors'] = st.slider('邻居数量', 1, 15, value=5, key='knn_n_neighbors')
        params['weights'] = st.selectbox('权重', ['uniform', 'distance'], key='knn_weights')
        # params['metric'] = st.selectbox('距离度量', ['minkowski', 'euclidean', 'manhattan'], key='knn_metric')
    elif clf_name == 'Random Forest':
        params['n_estimators'] = st.slider('决策树数量', 10, 300, value=100, key=f'{prefix}_rf_n_estimators')  # 默认值100
        params['max_depth'] = st.slider('最大深度', 2, 20, value=10, key=f'{prefix}_rf_max_depth')  # 默认值10
        params['max_features'] = st.select_slider('最大特征数', options=['sqrt', 'log2', 'auto'], value='sqrt', key=f'{prefix}_rf_max_features')  # 默认值'sqrt'
        params['min_samples_split'] = st.slider('最小分裂样本数', 2, 10, value=2, key=f'{prefix}_rf_min_samples_split')  # 默认值2
        params['min_samples_leaf'] = st.slider('最小叶子样本数', 1, 100, value=50, key=f'{prefix}_rf_min_samples_leaf')  # 默认值50
        params['random_state'] = st.slider('随机种子', 1, 100, value=5, key=f'{prefix}_rf_random_state')  # 默认值5
    elif clf_name == 'Logistic Regression':
        params['C'] = st.slider('正则化强度', 0.01, 10.0, value=1.0, key=f'{prefix}_lr_C')
        params['penalty'] = st.selectbox('正则化类型', ['l2','l1'], key=f'{prefix}_lr_penalty')
        params['solver'] = st.selectbox('优化器', ['lbfgs', 'liblinear', 'saga'], key=f'{prefix}_lr_solver')
        params['max_iter'] = st.slider('最大迭代次数', 100, 2000, value=1000, key=f'{prefix}_lr_max_iter')
    elif clf_name == 'Decision Tree':
        params['max_depth'] = st.slider('树的最大深度', 1, 20, value=10, key='dt_max_depth')
        params['min_samples_split'] = st.slider('最小分裂样本数', 2, 10, value=2, key='dt_min_samples')
        params['criterion'] = st.selectbox('分裂标准', ['gini', 'entropy'], key='dt_criterion')
    elif clf_name == 'AdaBoost':
        n_estimators = st.slider('n_estimators', 10, 300, key='ada_estimators')
        learning_rate = st.slider('learning_rate', 0.01, 1.0, key='ada_lr')
        params['n_estimators'] = n_estimators
        params['learning_rate'] = learning_rate
    elif clf_name == 'GradientBoosting':
        n_estimators = st.slider('n_estimators', 10, 300, 100, key='gb_estimators')
        learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, key='gb_lr')
        min_samples_split = st.slider('min_samples_split', 2, 20, 2, key='gb_min_split')
        min_samples_leaf = st.slider('min_samples_leaf', 1, 100, 50, key='gb_min_leaf')
        max_features = st.selectbox('max_features', ['sqrt', 'log2', None], key='gb_max_features')
        subsample = st.slider('subsample', 0.1, 1.0, 0.7, key='gb_subsample')
        random_state = st.number_input('random_state', value=5, key='gb_random_state')
        
        params['n_estimators'] = n_estimators
        params['learning_rate'] = learning_rate
        params['min_samples_split'] = min_samples_split
        params['min_samples_leaf'] = min_samples_leaf
        params['max_features'] = max_features
        params['subsample'] = subsample
        params['random_state'] = random_state
    elif clf_name == 'XGBoost':
        n_estimators = st.slider('n_estimators', 10, 300, value=100, key='xgb_estimators')  # 默认值设为100
        learning_rate = st.slider('learning_rate', 0.01, 1.0, value=0.01, key='xgb_lr')  # 默认值设为0.01
        max_depth = st.slider('max_depth', 3, 10, value=5, key='xgb_depth')  # 默认值设为5
        min_child_weight = st.slider('min_child_weight', 1, 10, value=1, key='xgb_min_child_weight')  # 默认值设为1
        subsample = st.slider('subsample', 0.5, 1.0, value=0.8, step=0.1, key='xgb_subsample')  # 默认值设为0.8
        colsample_bytree = st.slider('colsample_bytree', 0.5, 1.0, value=1.0, step=0.1, key='xgb_colsample_bytree')  # 默认值设为1.0
        gamma = st.slider('gamma', 0.0, 1.0, value=0.0, step=0.1, key='xgb_gamma')  # 默认值设为0.0
        reg_lambda = st.slider('reg_lambda (L2 regularization)', 0.0, 2.0, value=1.0, step=0.1, key='xgb_reg_lambda')  # 默认值设为1.0
        reg_alpha = st.slider('reg_alpha (L1 regularization)', 0.0, 2.0, value=0.2, step=0.1, key='xgb_reg_alpha')  # 默认值设为0.2

        params['n_estimators'] = n_estimators
        params['learning_rate'] = learning_rate
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        params['gamma'] = gamma
        params['reg_lambda'] = reg_lambda  # 添加L2正则化参数
        params['reg_alpha'] = reg_alpha  # 添加L1正则化参数
    elif clf_name == 'LightGBM':
        n_estimators = st.slider('n_estimators', 50, 300, value=100, key='lgb_n_estimators')
        learning_rate = st.slider('学习率', 0.01, 0.3, value=0.1, key='lgb_lr')
        max_depth = st.slider('最大深度', 3, 10, value=5, key='lgb_depth')
        num_leaves = st.slider('叶子节点数', 15, 127, value=31, key='lgb_leaves')
        min_child_samples = st.slider('最小子样本数', 20, 100, value=20, key='lgb_min_child_samples')
        params['n_estimators'] = n_estimators
        params['learning_rate'] = learning_rate
        params['max_depth'] = max_depth
        params['num_leaves'] = num_leaves
        params['min_child_samples'] = min_child_samples

    return params

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(probability=True,C=params.get('C', 1.0), kernel=params.get('kernel', 'rbf'), gamma=params.get('gamma', 'scale'))
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params.get('n_neighbors', 5), weights=params.get('weights', 'uniform'), metric=params.get('metric', 'minkowski'))
    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),  # 默认值100
            max_depth=params.get('max_depth', 10),  # 默认值10
            max_features=params.get('max_features', 'sqrt'),  # 默认值'sqrt'
            min_samples_split=params.get('min_samples_split', 2),  # 默认值2
            min_samples_leaf=params.get('min_samples_leaf', 50),  # 默认值50
            random_state=params.get('random_state', 5)  # 默认值5
        )
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params.get('C1', 1.0), penalty=params.get('penalty', 'l2'), solver=params.get('solver', 'lbfgs'), max_iter=params.get('max_iter', 1000), random_state=42)
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=params.get('max_depth', 10), min_samples_split=params.get('min_samples_split', 2), criterion=params.get('criterion', 'gini'), random_state=42)
    elif clf_name == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 1.0))
    elif clf_name == 'GradientBoosting':
        clf = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 50),
            max_features=params.get('max_features', 'sqrt'),
            subsample=params.get('subsample', 0.7),
            random_state=params.get('random_state', 5)
        )
    elif clf_name == 'XGBoost':
        clf = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),  # 默认值100
            learning_rate=params.get('learning_rate', 0.01),  # 默认值0.01
            max_depth=params.get('max_depth', 5),  # 默认值5
            min_child_weight=params.get('min_child_weight', 1),  # 默认值1
            subsample=params.get('subsample', 0.8),  # 默认值0.8
            colsample_bytree=params.get('colsample_bytree', 1.0),  # 默认值1.0
            gamma=params.get('gamma', 0.0),  # 默认值0.0
            reg_lambda=params.get('reg_lambda', 1.0),  # 默认值1.0（L2正则化）
            reg_alpha=params.get('reg_alpha', 0.2)  # 默认值0.2（L1正则化）
        )
    elif clf_name == 'LightGBM':
        clf = LGBMClassifier(n_estimators=params.get('n_estimators', 100), learning_rate=params.get('learning_rate', 0.1), max_depth=params.get('max_depth', 5), num_leaves=params.get('num_leaves', 31), min_child_samples=params.get('min_child_samples', 20), random_state=42)

    return clf