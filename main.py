from datetime import datetime
import json
import pickle
import rasterio
import streamlit as st 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import folium
from folium import plugins
from sklearn.metrics import roc_curve, auc
from streamlit_folium import folium_static
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns   
from sklearn.model_selection import GridSearchCV
from utils import get_dataset
from data_preprocessing import clean_outliers, get_scaler, standardize_data, standardize_datayc
from model_training import add_parameter_ui, get_classifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
import json
# from converted_script import read_and_process_bands
from yuce import *
import tempfile
from rasterio.plot import show

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
# 在文件最开始，导入语句之后添加
if 'data' not in st.session_state:
    # 初始化空的DataFrame
    st.session_state.data = pd.DataFrame()
X_scaled = pd.DataFrame()
y_test=pd.DataFrame()

with st.sidebar:
    choice=st.radio("Navigation",['数据预处理','模型训练','模型评估','模型预测'])
    # classifier_name = st.selectbox('选择模型',('KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Decision Tree', 
    #  'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'))
# dataset_name = st.sidebar.selectbox(
#     'Select Dataset',
#     ('Custom Dataset', 'Iris', 'Wine', 'Breast Cancer')
# )



# 写入数据集名称


# 如果是首次加载数据，将数据存入session state

# 在侧边栏中添加遥感影像上传功能
# with st.sidebar:
#     st.markdown("### 上传遥感影像文件")
#     uploaded_raster_file = st.file_uploader("选择文件", type=['tif', 'jpg', 'png', 'jpeg'])

#     if uploaded_raster_file is not None:
#         try:
#             # 你可以在这里添加读取和处理遥感影像的代码
#             # 例如，使用 rasterio 或其他库读取影像文件
#             import rasterio
#             from rasterio.plot import show
            
#             raster_data = rasterio.open(uploaded_raster_file)
#             st.success('文件上传成功!')
            
#             # 显示遥感影像
#             st.image(raster_data.read(1), caption='上传的遥感影像')
            
#         except Exception as e:
#             st.error(f'文件上传失败: {str(e)}')
# Add data input section in sidebar
with st.sidebar:
    st.markdown("### 上传数据文件")
    uploaded_file = st.file_uploader("上传栅格数据文件 (tif 格式)", type=["tif"])
    if uploaded_file is not None:
            # 将上传的文件保存到临时路径
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
    uploaded_file = st.file_uploader("选择样点文件", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                new_data = pd.read_csv(uploaded_file)
            else:
                new_data = pd.read_excel(uploaded_file)
                
            # Append new data to existing DataFrame
            st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
            X = st.session_state.data.iloc[:, :-1]
            st.success('文件上传成功!')
            
        except Exception as e:
            st.error(f'文件上传失败: {str(e)}')
if uploaded_file is None:
    st.info('请上传数据文件')
    st.stop()

# intermediate_shapefile_path = r"F:\rzf\intermediate_points10.shp"
# excel_to_shapefile(new_data, intermediate_shapefile_path)

gdf=excel_to_shapefile(new_data)

# transformed_shapefile_path = r"F:\rzf\transformed_points10.shp"  # 转换后的 Shapefile 路径
# transform_shapefile_crs(intermediate_shapefile_path, tmp_file.name, transformed_shapefile_path)

transformed_gdf=transform_shapefile_crs(gdf,tmp_file.name)

# final_shapefile_path = r"F:\rzf\points_with_band_values10.shp"  # 最终 Shapefile 路径
# extract_raster_values(transformed_shapefile_path, tmp_file.name, final_shapefile_path)

extracted_gdf = extract_raster_values(transformed_gdf, tmp_file.name)


 # 输出 Excel 文件路径
output_excel_path = r"F:\rzf\output_data11.xlsx"  # 替换为你的输出路径
# 调用函数进行转换
# shapefile_to_excel(extracted_gdf, output_excel_path)
excel_bytes = shapefile_to_excel(extracted_gdf)

# df = pd.read_excel(excel_bytes)

# temp_file_path = read_and_extract_dn_values(tmp_path, new_data)
# print(tmp_path)
temp_file_path = pd.read_excel(excel_bytes)
# st.write(temp_file_path)
X, y, data = get_dataset(temp_file_path)
X = pd.DataFrame(X)
data = pd.DataFrame(data)
temp_fil = pd.read_excel(output_excel_path)

# st.map(temp_fil)
# st.map(new_data, zoom=6, use_container_width=True)
def save_model(clf, classifier_name, params, acc, selected_features_ml, save_dir='models/', model_name=None, version='v1'):
    try:
        # 确保目录存在
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 构建默认文件名
        if not model_name:
            model_name = f'{classifier_name}_model'

        # 构建完整的保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(save_dir, f"{model_name}_{version}_{timestamp}.joblib")

        # 保存模型
        joblib.dump(clf, full_path)

        # 保存模型相关信息
        model_info = {
            'model_name': classifier_name,
            'parameters': params,
            'accuracy': acc,
            'features': selected_features_ml,
            'save_time': timestamp
        }

        # 保存信息文件
        info_path = os.path.join(save_dir, f"{model_name}_{version}_{timestamp}_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)

        return full_path, info_path

    except Exception as e:
        st.error(f"保存模型时发生错误: {str(e)}")
        return None, None



st.title('融雪洪水预测')
#####################################################################################3
# 标准化方法选择

    

if choice=='数据预处理':
        
        tab1, tab2,tab3,tab4 = st.tabs(["异常值处理","相关性分析","数据标准化", "数据预览"])
        # 设置页面标题,放在最开始位置
        # st.write(page_title="栅格和矢量图显示")

        # 读取栅格图像
        raster_path = r"F:\rzf\jfjCompositeBands.tif"
        with rasterio.open(raster_path) as raster:
            raster_img = raster.read(1)

        # 读取矢量图层
        points_path =extracted_gdf
        points = transformed_gdf

        # # 显示栅格图
        # st.subheader("栅格图像")
        # fig1, ax1 = plt.subplots(figsize=(12, 8))
        # show(raster_img, ax=ax1)
        # st.pyplot(fig1)

        # # 显示矢量图
        # st.subheader("矢量图层") 
        # fig2, ax2 = plt.subplots(figsize=(12, 8))
        # points.plot(ax=ax2, color='red', markersize=10)
        # st.pyplot(fig2)

        # 叠加显示
        # st.subheader("叠加显示")
        # fig3, ax3 = plt.subplots(figsize=(12, 8))
        # show(raster_img, ax=ax3)  
        # points.plot(ax=ax3, color='red', markersize=10)
        # st.pyplot(fig3)
  

 

        with tab1:
            st.title('异常值处理')
            target_columns = st.multiselect(
            '选择要分析的特征',
            options=X.columns.tolist(),
            default=X.columns.tolist(),
            key='multiselect_target_columns'
            )
            def replace_with_mean(df):
                    return df.fillna(df.mean())

            def replace_with_median(df):
                    return df.fillna(df.median())

            def drop_outliers(df):
                    return df.dropna()
                # 用户选择处理方法
            method = st.radio('选择异常值处理方法:', ('median', 'mean', 'Drop Outliers'))

                # 根据用户选择处理数据
            df_cleaned = clean_outliers(X[target_columns],method)
            remaining_features = X.columns.difference(target_columns)
            df_cleaneds = pd.concat([df_cleaned, X[remaining_features]], axis=1)
            # 显示处理后的数据
            st.write('Original Data:')
            st.write(X)
            st.write('Handled Data:')
            st.write(df_cleaneds)
            # df_cleaned = X

        with tab2:
            st.title('相关性分析')
            
            # 用户选择要分析的特征
            selected_features = st.multiselect(
                        '选择要分析的特征',
                        options=df_cleaneds.columns.tolist(),
                        default=df_cleaneds.columns.tolist(),
                        key='multiselect_selected_features'
                        )
            # 只计算用户选择的特征之间的相关性

            correlation_matrix = df_cleaneds[selected_features].corr(method='pearson')

            # 相关性分析方法选择
            correlation_method = st.selectbox(
                '选择相关性分析方法',
                ('Pearson', 'Spearman', 'Kendall')
            )

            # 根据选择的方法计算相关系数矩阵
            if correlation_method == 'Pearson':
                correlation_matrix = df_cleaneds[selected_features].corr(method='pearson')
            elif correlation_method == 'Spearman':
                correlation_matrix = df_cleaneds[selected_features].corr(method='spearman')
            elif correlation_method == 'Kendall':
                correlation_matrix = df_cleaneds[selected_features].corr(method='kendall')
            

            # 显示相关系数矩阵
            st.write("特征之间的皮尔逊相关系数矩阵:")
            st.write(correlation_matrix)
            fig, ax = plt.subplots(figsize=(20, 24))
            correlation_threshold = st.slider('选择相关性阈值', 0.0, 1.0, value=0.5)
            sns.heatmap(correlation_matrix, 
                                        annot=True,  
                                        cmap='coolwarm',
                                        center=0,
                                        square=True,
                                        fmt='.2f',
                                        linewidths=1,  # 增加网格线宽度
                                        annot_kws={'size': 8},  # 减小数字标注大小
                                        cbar_kws={'shrink': .8})  # 调整颜色条大小

            plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
            plt.yticks(rotation=0)
            st.write("特征之间的皮尔逊相关系数矩阵热力图:")
            st.pyplot(fig)
            # 显示原始DataFrame
            st.subheader('原始数据')
            df_cleaneds = pd.DataFrame(df_cleaneds[selected_features])
            st.write(df_cleaneds)
                # 计算特征与目标变量的相关性
            target_variable = 'Level'  # 假设目标变量名为'Level'
            correlation_with_target = df_cleaneds[selected_features].corrwith(data[target_variable]).abs().sort_values(ascending=False)
            
            # 显示特征与目标变量的相关性
            st.write(f"特征与目标变量'{target_variable}'的相关性:")
            
            # 可视化特征与目标变量的相关性
            fig, ax = plt.subplots(figsize=(10, 6))
            correlation_with_target.plot(kind='bar', ax=ax)
            plt.title(f"特征与目标变量'{target_variable}'的相关性")
            plt.xlabel('特征')
            plt.ylabel('绝对相关系数')
            st.pyplot(fig)

            # 计算相关性大于阈值的特征对
            high_correlation_pairs = []
            for col1 in correlation_matrix.columns:
                for col2 in correlation_matrix.columns:
                    if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > correlation_threshold:
                        high_correlation_pairs.append((col1, col2))

            # 删除相关性高的特征
            features_to_remove = set()
            for pair in high_correlation_pairs:
                # 假设我们总是移除第二列特征
                features_to_remove.add(pair[1])

            # 从DataFrame中删除选定的特征
            selected = [x for x in selected_features if x not in features_to_remove]

            # 显示删除后的DataFrame
            st.subheader('删除高相关性特征后的数据')
            df_cleaned = df_cleaneds[selected]
            st.write(df_cleaned)

                # 调整热力图参数
            
            
            



            with tab3:
                st.title('数据标准化')
                # scaler_name = st.sidebar.selectbox(
                #     '选择标准化方法',
                #     ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler')
                # )
                # # 用户选择要标准化的特征
                # features_to_scale = st.multiselect(
                #     '选择要标准化的特征',
                #     options=df_cleaned.columns.tolist(),
                #     default=df_cleaned.columns.tolist()
                # )
                # # 用户选择标准化方法
                # method = st.radio('Choose a method to handle outliers:', ('StandardScaler', 'MinMaxScaler', 'RobustScaler','MaxAbsScaler'))
                # scaler = get_scaler(method)
                # X_scaled = standardize_data(df_cleaned, scaler,features_to_scale)
                # # X_scaled['FeatureName'] = features_to_scale
                # # 显示标准化后的数据
                # st.write(X_scaled)
                # scaler_name = st.sidebar.selectbox(
                #     '选择标准化方法',
                #     ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler')
                # )
                # 用户选择要标准化的特征
                features_to_scale = st.multiselect(
                    '选择要标准化的特征',
                    options=df_cleaneds.columns.tolist(),
                    default=df_cleaned.columns.tolist(),
                    key='multiselect_features_to_scale'
                )

                # 用户选择标准化方法
                method = st.radio('选择标准化方法:', ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'))
                scaler = get_scaler(method)
                
                # 标准化选定的特征
                X_scaled = standardize_data(df_cleaneds, scaler,features_to_scale)                
                # 将未被选择的特征重新合并回数据集
                unscaled_features = df_cleaned.columns.difference(features_to_scale)
                X_final = pd.concat([X_scaled, df_cleaned[unscaled_features]], axis=1)
                
                # 显示标准化后的数据
                st.write('标准化后的数据:')
                st.write(X_final)
                
                # 将标准化后的数据存储在会话状态中
                st.session_state['X_scaled'] = X_final  # 将X_scaled存储在会话状态中
                
            with tab4:
                 st.title('数据预览')
                 st.write(X_scaled)
                #  colors = ['red' if Level == 1 else 'blue' for Level in data['Level']]
                # # 创建一个地图
                #  st.title('地图')
                #  fig, ax = plt.subplots()
                #  for i in range(len(df_cleaned)):
                #     ax.scatter(df_cleaned['lon'][i], df_cleaned['lat'][i], color=colors[i], s=10)

                # # 显示地图
                #  st.pyplot(fig)

                # 使用Streamlit的map组件也可以实现类似的功能，但需要Streamlit的版本支持
                # 以下是一个使用Streamlit map组件的示例
                 st.map(new_data, zoom=6, use_container_width=True)
                 st.session_state['X_scaled'] = X_scaled  # 将X_scaled存储在会话状态中

if choice=='模型训练':
        st.title('特征选择')
        if 'X_scaled' in st.session_state:
            X_scaled = st.session_state['X_scaled']

            
    # 用户选择要分析的特征
        selected_features_ml = st.multiselect(
            '选择用于机器学习的特征',
            options=X_scaled.columns.tolist(),
            default=X_scaled.columns.tolist()
        )
        test_size = st.slider('测试集大小', 0.1, 0.9, 0.2)
        tab1, tab2,tab3 = st.tabs(["机器学习","stacking","深度学习"])
        
        with tab1:
            st.title('模型调参')

            classifier_name = st.selectbox('选择模型', ('KNN', 'SVM', 'Random Forest', 'Logistic Regression', 'Decision Tree', 
                                                        'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM'))
            X_selected = X_scaled[selected_features_ml]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=1234)

            use_grid_search = st.checkbox('使用网格搜索优化参数')
            
            if use_grid_search:
                # 为不同模型定义参数网格
                param_grids = {
                    'KNN': {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    },
                    'SVM': {
                        'C': [0.1, 1],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto', 0.1, 1]
                    },
                    'Random Forest': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'max_features': ['sqrt', 'log2'],
                        'random_state': [5],
                        'min_samples_leaf': [1,100]

                    },
                    'Logistic Regression': {
                        'C': [0.01, 0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    },
                    'Decision Tree': {
                        'max_depth': [5, 10, 15, 20],
                        'min_samples_split': [2, 5, 10],
                        'criterion': ['gini', 'entropy']
                    },
                    'LightGBM': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [31, 63, 127],
                        'min_child_samples': [20, 50, 100]
                    },
                    'AdaBoost': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0],
                        'algorithm': ['SAMME', 'SAMME.R']
                    },
                    'XGBoost': {
                        'n_estimators': [100, 200, 300],  # 默认范围保持不变
                        'learning_rate': [0.01, 0.1, 0.3],  # 默认范围保持不变
                        'max_depth': [3, 5, 7],  # 默认范围保持不变
                        'min_child_weight': [1, 3, 5],  # 默认范围保持不变
                        'subsample': [0.8],  # 固定为0.8，与照片中的建议一致
                        'colsample_bytree': [0.8, 0.9, 1.0],  # 默认范围保持不变
                        'gamma': [0, 0.1, 0.2],  # 默认范围保持不变
                        'reg_lambda': [1,5],  # L2正则化参数，固定为1，与照片中的建议一致
                        'reg_alpha': [0, 0.2]  # L1正则化参数，范围为[0, 0.2]，与照片中的建议一致
                    },
                    'GradientBoosting': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.3],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 50],  # 修改为包含50，与代码中的默认值一致
                        'max_features': ['sqrt', 'log2'],
                        'subsample': [0.7, 0.8, 0.9],  # 修改为包含0.7，与代码中的默认值一致
                        'random_state': [5]  # 添加random_state参数
                    }
                    }
            
                
                # 获取当前选择模型的参数网格
                current_param_grid = param_grids[classifier_name]
                
                # 创建分类器实例
                base_clf = get_classifier(classifier_name, {})
                
                # 创建GridSearchCV对象
                grid_search = GridSearchCV(
                    estimator=base_clf,
                    param_grid=current_param_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring='accuracy'
                )
            
                # 执行网格搜索
                with st.spinner('正在执行网格搜索...'):
                    grid_search.fit(X_train, y_train)
                
                # 显示最佳参数和得分
                st.write('最佳参数:', grid_search.best_params_)
                st.write('最佳交叉验证得分:', grid_search.best_score_)
                params=grid_search.best_params_
                # 使用最佳模型进行预测
                clf = grid_search.best_estimator_
                y_pred = clf.predict(X_test)
                st.session_state['best_params'] = grid_search.best_params_
                st.session_state['y_pred_tuned'] = y_pred
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['classifier_name'] = classifier_name

                st.session_state['y_score_tuned'] = clf.predict_proba(X_test)
                # if st.button('保存模型',key='save_single_model'):
                    # model_filename = st.text_input('请输入模型保存路径和文件名', 'model.joblib')
                    # if model_filename:
                    #     joblib.dump(clf, model_filename)
                    #     st.success(f"模型已保存为 {model_filename}")
                    #     st.session_state['model_filename'] = model_filename
                    # else:
                    #     st.warning("请输入有效的文件路径和文件名")
                st.subheader('模型保存')

                # 创建保存目录选项
                save_dir = st.text_input('请输入保存目录路径(留空则保存在当前目录)', 'models/')

                # 自定义文件名
                model_name = st.text_input('请输入模型文件名(不包含扩展名)', f'{classifier_name}_model')

                # 添加版本号选项
                version = st.text_input('版本号(可选)', 'v1')
                acc = accuracy_score(y_test, y_pred)
# 创建模型包，包含模型和元数据
                model_binary = pickle.dumps(clf)
                timestamp = datetime.now()                
                # 创建可序列化的元数据字典
                metadata = {
                    "model_type": classifier_name,
                    "parameters": grid_search.best_params_,
                    "features": list(selected_features_ml),
                    "accuracy": float(grid_search.best_score_),
                    "feature_count": len(selected_features_ml)
                }
                metadata_json = json.dumps(metadata, indent=4, ensure_ascii=False)
                # 创建下载按钮
                timestamp = datetime.now()
                col1, col2 = st.columns(2)

                with col1:
                    # 下载模型按钮
                    st.download_button(
                        label="下载模型文件",
                        data=model_binary,
                        file_name=f"{classifier_name}_model_{timestamp}.pkl",
                        mime="application/octet-stream"
                    )

                with col2:
                    # 下载元数据按钮
                    st.download_button(
                        label="下载模型元数据",
                        data=metadata_json,
                        file_name=f"{classifier_name}_metadata_{timestamp}.json",
                        mime="application/json"
                    )
            else:
                # 原有的参数选择和模型训练代码
                params = add_parameter_ui(classifier_name,'main')
                clf = get_classifier(classifier_name, params)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # 计算评估指标
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cc = np.corrcoef(y_test, y_pred)[0, 1]
                bias = np.mean(y_pred - y_test)
                nse = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

                # 创建图表
                fig, ax = plt.subplots(figsize=(8, 6))

                # 绘制散点图
                ax.scatter(y_test, y_pred, color='blue', label='Data Points')

                # 绘制对角线（理想情况）
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line')

                # 添加标注信息
                ax.text(0.1, 0.9, f'CC={cc:.3f}\nBias={bias:.3f}\nRMSE={rmse:.3f}\nNSE={nse:.3f}', 
                        fontsize=12, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

                # 设置标题和标签
                ax.set_title('Real Level vs Estimated Level (GBDT)', fontsize=16)
                ax.set_xlabel('Real Level', fontsize=14)
                ax.set_ylabel('Estimated Level', fontsize=14)

                # 添加图例
                ax.legend()

                # 显示网格
                ax.grid(True)

                # 在 Streamlit 中显示图表
                st.pyplot(fig)

                # 显示评估指标
                st.write(f"RMSE: {rmse:.3f}")
                st.write(f"Correlation Coefficient (CC): {cc:.3f}")
                st.write(f"Bias: {bias:.3f}")
                st.write(f"Nash-Sutcliffe Efficiency (NSE): {nse:.3f}")

                st.session_state['best_params'] = params
                st.session_state['y_pred_tuned'] = y_pred
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test
                st.session_state['y_score_tuned'] = clf.predict_proba(X_test)
                
                st.session_state['classifier_name'] = classifier_name
                # if st.button('保存模型',key='save_single_model'):
                #     model_filename = f"{classifier_name}_model.joblib"
                #     joblib.dump(clf, model_filename)
                #     st.success(f"模型已保存为 {model_filename}")
                #     st.session_state['model_filename'] = model_filename
                #     st.write(model_filename)
                    # 添加模型保存功能

                acc = accuracy_score(y_test, y_pred)
                # 添加保存按钮
                model_binary = pickle.dumps(clf)
                timestamp = datetime.now()                
                # 创建可序列化的元数据字典
                metadata = {
                    "model_type": classifier_name,
                    "parameters": params,
                    "features": list(selected_features_ml),
                    "accuracy": float(acc),
                    "feature_count": len(selected_features_ml)
                }
                metadata_json = json.dumps(metadata, indent=4, ensure_ascii=False)
                # 创建下载按钮
                timestamp = datetime.now()
                col1, col2 = st.columns(2)

                with col1:
                    # 下载模型按钮
                    st.download_button(
                        label="下载模型文件",
                        data=model_binary,
                        file_name=f"{classifier_name}_model_{timestamp}.pkl",
                        mime="application/octet-stream"
                    )

                with col2:
                    # 下载元数据按钮
                    st.download_button(
                        label="下载模型元数据",
                        data=metadata_json,
                        file_name=f"{classifier_name}_metadata_{timestamp}.json",
                        mime="application/json")

            st.session_state['training_method'] = 'tuned'
            # 计算准确率
            acc = accuracy_score(y_test, y_pred)
            st.write(f'Classifier = {classifier_name}')
            st.write(f'Accuracy = {acc:.2f}')

    # 在tab2（Stacking模型配置）中，训练完Stacking模型后，将预测结果、测试集等信息存储在st.session_state中
        with tab2:
            st.title('stacking')
            # 添加Stacking模型配置选项卡
            with st.expander("Stacking模型配置"):
                st.subheader('选择基础模型')
                base_models_available = [
                    ('knn', KNeighborsClassifier()),
                    ('svm', SVC(probability=True)),
                    ('Random Forest', RandomForestClassifier()),
                    ('Logistic Regression', LogisticRegression()),
                    ('Decision Tree', DecisionTreeClassifier()),
                    ('AdaBoost', AdaBoostClassifier()),
                    ('Gradient Boosting', GradientBoostingClassifier()),
                    ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
                    ('LightGBM', lgb.LGBMClassifier()),
                    ('CatBoost', CatBoostClassifier(verbose=0))
                ]
                base_model_choices = st.multiselect(
                    '选择基础模型',
                    options=[name for name, _ in base_models_available],
                    default=['knn', 'svm', 'Random Forest']
                )
                base_models = [(name, model) for name, model in base_models_available if name in base_model_choices]

                st.subheader('选择元模型')
                meta_model_name = st.selectbox(
                    '选择元模型',
                    ('Logistic Regression', 'Random Forest', 'SVM')
                )
                meta_model_params = add_parameter_ui(meta_model_name, 'stacking')
                meta_model = get_classifier(meta_model_name, meta_model_params)

                if st.button('训练Stacking模型'):
                    with st.spinner('训练Stacking模型...'):
                        stacking_clf = StackingClassifier(
                            estimators=base_models,
                            final_estimator=meta_model,
                            cv=5
                        )
                        stacking_clf.fit(X_train, y_train)
                        y_pred_stacking = stacking_clf.predict(X_test)
                        acc_stacking = accuracy_score(y_test, y_pred_stacking)
                        st.write(f'Stacking Classifier Accuracy = {acc_stacking:.2f}')
                        
                        # 将预测结果、测试集等信息存储在st.session_state中
                        st.session_state['y_pred_stacking'] = y_pred_stacking
                        st.session_state['y_test'] = y_test
                        st.session_state['X_test'] = X_test
                        st.session_state['y_score_stacking'] = stacking_clf.predict_proba(X_test)
                        st.session_state['base_model_choices'] = base_model_choices
                        st.session_state['meta_model_name'] = meta_model_name
                        st.session_state['training_method'] = 'stacking'
                        st.session_state['stacking_clf'] = stacking_clf
                        model_binary = pickle.dumps(stacking_clf)
                        timestamp = datetime.now()

                        # 创建可序列化的元数据字典
                        metadata = {
                            "model_type": "Stacking",
                            "base_models": base_model_choices,
                            "meta_model": meta_model_name,
                            "accuracy": float(acc_stacking),
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        metadata_json = json.dumps(metadata, indent=4, ensure_ascii=False)

                        # 创建下载按钮
                        col1, col2 = st.columns(2)

                        with col1:
                            # 下载模型按钮
                            st.download_button(
                                label="下载模型文件",
                                data=model_binary,
                                file_name=f"stacking_model_{timestamp}.pkl",
                                mime="application/octet-stream"
                            )

                        with col2:
                            # 下载元数据按钮
                            st.download_button(
                                label="下载模型元数据",
                                data=metadata_json,
                                file_name=f"stacking_metadata_{timestamp}.json",
                                mime="application/json"
                            )

        with tab3:
            st.title('深度学习')         
        st.session_state['X_scaled'] = X_scaled[selected_features_ml]
            

    # 在choice=='模型评估'部分，根据用户的训练选择来读取相应的结果
if choice=='模型评估':
    st.title('模型评估报告')

    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state['X_scaled']

    # 检查用户选择了哪种训练方法
    if 'training_method' in st.session_state:
        training_method = st.session_state['training_method']
    else:
        training_method = None

    if training_method == 'tuned':
        # 读取模型调参的结果
        if 'y_pred_tuned' in st.session_state:
            y_pred = st.session_state['y_pred_tuned']
        if 'y_test' in st.session_state:
            y_test = st.session_state['y_test']
        if 'X_test' in st.session_state:
            X_test = st.session_state['X_test']
        if 'y_score_tuned' in st.session_state:
            y_score = st.session_state['y_score_tuned']
        if 'best_params' in st.session_state:
            best_params = st.session_state['best_params']
        if 'classifier_name' in st.session_state:
            classifier_name = st.session_state['classifier_name']
        # 显示模型调参的结果
        st.write(best_params)
        st.write(f'使用的模型 = {classifier_name}')
        params_md = "\n".join([f"- **{key}**: {value}" for key, value in best_params.items()])
        st.write("模型使用的超参数:")
        st.write(params_md)
    elif training_method == 'stacking':
        # 读取Stacking模型的结果
        if 'y_pred_stacking' in st.session_state:
            y_pred = st.session_state['y_pred_stacking']
        if 'y_test' in st.session_state:
            y_test = st.session_state['y_test']
        if 'X_test' in st.session_state:
            X_test = st.session_state['X_test']
        if 'y_score_stacking' in st.session_state:
            y_score = st.session_state['y_score_stacking']
        if 'base_model_choices' in st.session_state:
                base_model_choices = st.session_state['base_model_choices']
        if 'meta_model_name' in st.session_state:
                meta_model_name = st.session_state['meta_model_name']
        # base_model_choices = pd.DataFrame(base_model_choices)
        # base_model_choices_md = "\n".join([f"- **{key}**: {value}" for key, value in base_model_choices.items()])
        # st.write("stacking 基础模型:")
        # st.write(base_model_choices_md)
        for model in base_model_choices:
            st.write(model)

        st.write(f'使用的元模型 = {meta_model_name}')
        st.write(f'使用的基础模型 = {base_model_choices}')

    else:
        st.warning('请先进行模型训练或Stacking模型训练。')
        st.stop()
    # if 'model_filename' in st.session_state:
    #     model_filename = st.session_state['model_filename']
    # st.write(f'模型保存路径 = {model_filename}')


    # 进行模型评估
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy = {acc:.2f}')

    # 使用PCA降维
    pca = PCA(2)
    X_projected = pca.fit_transform(X_scaled)

    # 绘制PCA散点图
    fig = plt.figure()
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.8, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    st.pyplot(fig)

    # 创建一个散点图，使用不同的颜色表示预测正确和预测错误的点
    # fig, ax = plt.subplots()
    # correct = np.where(y_pred == y_test)[0]
    # ax.scatter(X_test[correct, 0], X_test[correct, 1], 
    #             c='g', marker='o', label='Correct Predictions')
    # incorrect = np.where(y_pred != y_test)[0]
    # ax.scatter(X_test[incorrect, 0], X_test[incorrect, 1], 
    #             c='r', marker='x', label='Incorrect Predictions')
    # ax.legend()
    # ax.set_title('Predicted vs True Labels')
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # st.pyplot(fig)

    # 获取预测概率
    y_score = y_score

    # 将标签进行one-hot编码
    y_test_encoded = pd.get_dummies(y_test).values
    n_classes = y_test_encoded.shape[1]

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_encoded[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'类别 {i} 的ROC曲线 (AUC = {roc_auc[i]:0.2f})')

    # 添加对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('多分类ROC曲线')
    plt.legend(loc="lower right")

    # 在Streamlit中显示图表
    st.pyplot(fig)

    # 为每个类别画校准曲线
    plt.figure(figsize=(10, 6))
    for i in range(y_score.shape[1]):
        # 计算校准曲线
        prob_true, prob_pred = calibration_curve(y_test == i, y_score[:, i], n_bins=10)
        
        # 绘制校准曲线
        plt.plot(prob_pred, prob_true, marker='o', label=f'类别 {i}')

    # 添加对角线（代表完美校准）
    plt.plot([0, 1], [0, 1], 'k--', label='完美校准')

    # 设置图表属性
    plt.xlabel('预测概率')
    plt.ylabel('实际概率')
    plt.title('模型校准曲线')
    plt.legend()
    plt.grid(True)

    # 在Streamlit中显示图表
    st.pyplot(plt)

    # 存储评估结果到session_state
    st.session_state['y_pred'] = y_pred
    st.session_state['y_test'] = y_test
    st.session_state['X_test'] = X_test
    st.session_state['y_score'] = y_score
if choice == '模型预测':
    # 从session_state中获取预测结果
    tab1,tab2 = st.tabs(['数据预处理','模型加载'])
    if 'X_scaled' in st.session_state:
                X_scaled = st.session_state['X_scaled']
    with tab1:
        # uploaded_file = st.file_uploader("上传栅格数据文件 (CSV 格式)", type=["tif"])
        # # create_training_data(uploaded_file)
        # band_arrays, num_bands, metadata = read_and_process_bands(uploaded_file)

        uploaded_file = st.file_uploader("上传栅格数据文件 (CSV 格式)", type=["tif"])
        if uploaded_file is not None:
            # 将上传的文件保存到临时路径
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # 调用 read_and_process_bands 函数
            band_dicts, num_bands, metadata = read_and_process_bands(tmp_path)
            dfyc = pd.DataFrame({f'band_{i+1}': band_dicts.flatten() for i, band_dicts in enumerate(band_dicts)})
            # dfyc = pd.DataFrame(band_dicts)
            # dfyc['lon'] = dfyc['坐标'].apply(lambda x: x[0])
            # dfyc['lat'] = dfyc['坐标'].apply(lambda x: x[1])
            # dfyc = dfyc.rename(columns={f'band{i}': f'band_{i}' for i in range(1, num_bands+1)})
            # dfyc = dfyc.drop(columns=['坐标'])

            # 显示处理后的数据
            st.write('Original Data:')
            st.write(dfyc)


            ################################################################################################################

            # 将数据转换为DataFrame
            # band_arrayspd = pd.DataFrame(band_arrays.reshape(-1, num_bands))
            def replace_with_mean(df):
                    return df.fillna(df.mean())

            def replace_with_median(df):
                    return df.fillna(df.median())

            def drop_outliers(df):
                    return df.dropna()
                # 用户选择处理方法
            method = st.radio('选择异常值处理方法:', ('median', 'mean', 'Drop Outliers'))

                # 根据用户选择处理数据
            df_cleanedyc = clean_outliers(dfyc,method)
            # 显示处理后的数据
            st.write('Original Data:')
            st.write(dfyc)
            st.write('Handled Data:')
            st.write(df_cleanedyc)

            ################################################################################################################
#################################################################################################################
            method = st.radio('选择标准化方法:', ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'))
            scaler = get_scaler(method)
                
                # 标准化选定的特征
            X_scaledyc = standardize_datayc(dfyc, scaler)           

                # 显示标准化后的数据
            st.write('Standardized Data:')
            st.write(X_scaledyc)
            #显示标准化之前的数据
            st.write('Original Data:')    
            st.write(df_cleanedyc)
##################################################################################################################

        # 显示数据预处理的结果
        st.write(f'数据预处理的结果 = ')
    with tab2:
        # 显示模型预测
        load_method = st.radio(
            "选择模型加载方式",
            ["使用已训练模型", "加载本地模型"]
            )
        if load_method == "使用已训练模型":
                if 'model_filename' in st.session_state:
                    model_filename = st.session_state['model_filename']
                    try:
                        loaded_model = joblib.load(model_filename)
                        st.success(f"模型 {model_filename} 已成功加载")
                    except FileNotFoundError:
                        st.error(f"模型文件 {model_filename} 未找到")
                        st.stop()
                else:
                    st.error("未找到模型名称")
                    st.stop()
        elif load_method == "加载本地模型":
            # 添加本地模型文件上传功能
                uploaded_model = st.file_uploader("上传模型文件", type=['joblib', 'pkl', 'pickle'])
            
                if uploaded_model:
                        try:
                        # 保存上传的模型文件到临时目录
                            temp_path = f"temp_models/{uploaded_model.name}"
                            os.makedirs("temp_models", exist_ok=True)
                            
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_model.getbuffer())
                            
                            # 加载模型
                            loaded_model = joblib.load(temp_path)
                            st.success(f"本地模型 {uploaded_model.name} 已成功加载")
                        
                        except Exception as e:
                            st.error(f"加载模型时发生错误: {str(e)}")
                            st.stop()
            # # 加载保存的模型
            # if 'model_filename' in st.session_state:
            #     model_filename = st.session_state['model_filename']
            #     # model_filename = f"{model_filename}_model.joblib"
            #     # clf = joblib.load(model_filename)
            #     try:
            #         loaded_model = joblib.load(model_filename)
            #         st.success(f"模型 {model_filename} 已成功加载")
            #     except FileNotFoundError:
            #         st.error(f"模型文件 {model_filename} 未找到")
            #         st.stop()
                else:
                                    st.error("未找到模型名称")
                                    st.stop()
                # 假设你有一个新的数据集需要进行预测
                # 这里假设你已经定义了clean_outliers和standardize_data函数
                # 并且知道使用了哪种标准化方法
        elif load_method is None:
                st.error("未训练模型")
                st.stop()
        if 'X_scaled' in st.session_state:
                X_scaled = st.session_state['X_scaled']

                # new_data_cleaned = clean_outliers(X_scaled, method='median')  # 根据实际情况调整方法
                # scaler = get_scaler('StandardScaler')  # 根据实际情况调整标准化方法
                # new_data_scaled = standardize_data(new_data_cleaned, scaler, new_data_cleaned.columns.tolist())
#                 X_array = np.random.rand(height * width, 11)  # 示例数据，形状为 (样本数, 特征数)

# # 调用 predict_flood_risk 函数
#                 Y_predict_proba, Y_predict, flood_probabilities_reshaped = predict_flood_risk(
#                     loaded_model, X_array, reference_raster_path, output_file, height, width
#                 )
                # band_arrays, num_bands, metadata = read_and_process_bands(image_path)

                # # 打印结果
                # print("预测概率:", Y_predict_proba)
                # print("预测结果:", Y_predict)
                # print("洪水概率的二维数组形状:", flood_probabilities_reshaped.shape)


                # 进行预测
                # predictions = loaded_model.predict(X_scaled)
                # probabilities = loaded_model.predict_proba(X_scaled)
                # st.write('预测结果:', predictions,data)
                # output_file=r"F:\rzf3.tif"

                # height, width = 477, 531 
                # Y_predict_proba, Y_predict, flood_probabilities_reshaped = predict_flood_risk(
                #     loaded_model, X_scaledyc, tmp_path, output_file, height, width
                # )

                # # 存储预测结果到session_state
                # st.session_state['y_pred'] = predictions
                # st.session_state['y_prob'] = probabilities
                # predictions = pd.DataFrame(predictions)
                # prediction = pd.concat([predictions, data[['lat', 'lon']]], axis=1)

                # # 输出预测结果
                # st.write('预测结果:', predictions,'原始结果',X_scaled,prediction)
                # st.write('预测概率:', probabilities)
                # st.map(prediction, zoom=6, use_container_width=True)
                ###############################################################################################
                # 添加一个按钮
                if st.button('运行预测'):
                    # 假设 X_scaledyc 是一个已经定义好的变量
                    x = X_scaledyc.values

                    # 定义图像的高度和宽度
                    height, width = 477, 531

                    # 预测洪水风险
                    Y_predict_proba, Y_predict, flood_probabilities_reshaped,raster_data,raster_dataset= predict_flood_risk(
                        loaded_model, x, tmp_path,height, width
                    )

                    # 打开栅格文件
                    # with rasterio.open(raster_dataset) as src:
                    #     flood_classes = src.read(1)  # 读取第一个波段

                    #     # 使用自然断点法进行分类
                    #     # 计算自然断点
                    #     thresholds = np.percentile(flood_classes[flood_classes > 0], [0, 20, 40, 60, 80, 100])
                    #     classified = np.digitize(flood_classes, thresholds)  # 将概率分为5类

                    #     # 创建图形
                    #     plt.figure(figsize=(12, 8))
                        
                    #     # 显示分类后的栅格数据
                    #     im = plt.imshow(classified, cmap='coolwarm', alpha=0.5)  # 使用 imshow 显示分类图
                    #     plt.colorbar(im, label='洪水风险等级 (1: 较低风险, 5: 高风险)')
                        
                    #     # 添加标题
                    #     plt.title('融雪型洪水风险分类图')
                        
                    #     # 显示图像
                    #     plt.axis('off')  # 关闭坐标轴
                    #     st.pyplot(plt.gcf())  # 使用 st.pyplot() 显示当前图形  

                    # st.title("洪水风险分类可视化")

                    flood_classes = flood_probabilities_reshaped  # 读取第一个波段

                    # 使用自然断点法进行分类
                    # 计算自然断点
                    thresholds = np.percentile(flood_classes[flood_classes > 0], [0, 20, 40, 60, 80, 100])
                    classified = np.digitize(flood_classes, thresholds)  # 将概率分为5类

                    # 创建图形
                    plt.figure(figsize=(12, 8))

                    # 显示分类后的栅格数据
                    im = plt.imshow(classified, cmap='coolwarm', alpha=0.5)  # 使用 imshow 显示分类图
                    plt.colorbar(im, label='洪水风险等级 (1: 较低风险, 5: 高风险)')
                    title="伊犁地区洪水风险分类图"
                    # 添加标题
                    plt.title(title)

                    # 显示图像
                    plt.axis('off')  # 关闭坐标轴
                    st.pyplot(plt.gcf())  # 使用 st.pyplot() 显示当前图形


                    st.download_button(
                        label="下载 TIFF 文件",
                        data=raster_data,
                        file_name="image.tif",
                        mime="image/tiff"
                    )
                else:
                    st.write("点击按钮以运行预测")    
        else:
                # st.error("未找到预处理后的数据")
                default_output_path = r"F:\rzf4.tif"

                # 用户自定义输出路径和文件名
                output_file = st.text_input('请输入保存目录路径及文件名（留空则使用默认路径）', default_output_path)

                # 添加一个按钮
#                 if st.button('运行预测'):
#                     # 假设 X_scaledyc 是一个已经定义好的变量
#                     x = X_scaledyc.values

#                     # 定义图像的高度和宽度
#                     height, width = 477, 531

#                     # 预测洪水风险
#                     Y_predict_proba, Y_predict, flood_probabilities_reshaped = predict_flood_risk(
#                         loaded_model, x, tmp_path, output_file, height, width
#                     )

#                     import streamlit as st

# # 读取 TIFF 文件的二进制数据

# # 提供下载按钮
#                     st.download_button(
#                         label="下载 TIFF 文件",
#                         data=flood_probabilities_reshaped,
#                         file_name="image.tif",
#                         mime="image/tiff"
#                     )

#                     # 打开栅格文件
#                     with rasterio.open(output_file) as src:
#                         flood_classes = src.read(1)  # 读取第一个波段

#                         # 使用自然断点法进行分类
#                         # 计算自然断点
#                         thresholds = np.percentile(flood_classes[flood_classes > 0], [0, 20, 40, 60, 80, 100])
#                         classified = np.digitize(flood_classes, thresholds)  # 将概率分为5类

#                         # 创建图形
#                         plt.figure(figsize=(12, 8))
                        
#                         # 显示分类后的栅格数据
#                         im = plt.imshow(classified, cmap='coolwarm', alpha=0.5)  # 使用 imshow 显示分类图
#                         plt.colorbar(im, label='洪水风险等级 (1: 较低风险, 5: 高风险)')
                        
#                         # 添加标题
#                         plt.title('融雪型洪水风险分类图')
                        
#                         # 显示图像
#                         plt.axis('off')  # 关闭坐标轴
#                         st.pyplot(plt.gcf())  # 使用 st.pyplot() 显示当前图形
#                 else:
                st.write("点击按钮以运行预测")
                st.stop()
