from io import BytesIO
import shutil
import tempfile
import warnings
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import cv2
import keras 
from keras.utils import normalize
from keras import *
import pandas as pd
from keras.models import load_model
from osgeo import gdal
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np # type: ignore
from geocube.api.core import make_geocube
import rasterio as rio
from rasterio.mask import mask
import numpy as np
from shapely import Point
from shapely.geometry import box, Polygon
from rasterio.features import shapes
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import glob
from osgeo import gdal, ogr
from rasterio.transform import from_origin
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
from openpyxl import load_workbook 
from rasterio.transform import rowcol
##################################################################################################################
# def excel_to_shapefile(excel_path, output_shapefile_path, lon_col='lon', lat_col='lat'):
#     """
#     将 Excel 表格中的经纬度数据转换为点 Shapefile 文件，并设置坐标系为 WGS 1984。
    
#     参数:
#         excel_path (str): Excel 文件路径。
#         output_shapefile_path (str): 输出的 Shapefile 文件路径。
#         lon_col (str): 经度列名，默认为 'lon'。
#         lat_col (str): 纬度列名，默认为 'lat'。
#     """
#     # 读取 Excel 文件
#     df = excel_path

#     # 检查经度和纬度列是否存在
#     if lon_col not in df.columns or lat_col not in df.columns:
#         raise ValueError(f"Excel 文件中必须包含 '{lon_col}' 和 '{lat_col}' 列")

#     # 创建几何点
#     geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

#     # 创建 GeoDataFrame，并设置坐标系为 WGS 1984 (EPSG:4326)
#     gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

#     # 保存为 Shapefile
#     gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
#     print(f"Shapefile 已保存到: {output_shapefile_path}")
def excel_to_shapefile(excel_path, lon_col='lon', lat_col='lat'):
    """
    将 Excel 表格中的经纬度数据转换为点 GeoDataFrame，并设置坐标系为 WGS 1984。
    
    参数:
        excel_path (str): Excel 文件路径。
        lon_col (str): 经度列名，默认为 'lon'。
        lat_col (str): 纬度列名，默认为 'lat'。
    
    返回:
        gdf (GeoDataFrame): 包含几何点的 GeoDataFrame，坐标系为 WGS 1984 (EPSG:4326)。
    """
    # 读取 Excel 文件
    df =excel_path

    # 检查经度和纬度列是否存在
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"Excel 文件中必须包含 '{lon_col}' 和 '{lat_col}' 列")

    # 创建几何点
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

    # 创建 GeoDataFrame，并设置坐标系为 WGS 1984 (EPSG:4326)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 返回 GeoDataFrame
    return gdf
# Step 2: 将 Shapefile 的地理坐标系转换为栅格图的投影坐标系
# def transform_shapefile_crs(shapefile_path, raster_path, output_shapefile_path):
#     """
#     将 Shapefile 的地理坐标系转换为栅格图的投影坐标系。
    
#     参数:
#         shapefile_path (str): 输入的 Shapefile 文件路径。
#         raster_path (str): 遥感影像文件路径。
#         output_shapefile_path (str): 输出的 Shapefile 文件路径。
#     """
#     # 读取 Shapefile 文件
#     points = gpd.read_file(shapefile_path)

#     # 读取栅格文件的投影坐标系
#     with rasterio.open(raster_path) as src:
#         raster_crs = src.crs

#     # 将 Shapefile 的坐标系转换为栅格图的投影坐标系
#     points = points.to_crs(raster_crs)

#     # 保存转换后的 Shapefile
#     points.to_file(output_shapefile_path, driver='ESRI Shapefile')
#     print(f"转换后的 Shapefile 已保存到: {output_shapefile_path}")


def transform_shapefile_crs(shapefile_path, raster_path):
    """
    将 Shapefile 的地理坐标系转换为栅格图的投影坐标系。
    
    参数:
        shapefile_path (str): 输入的 Shapefile 文件路径。
        raster_path (str): 遥感影像文件路径。
    
    返回:
        points (GeoDataFrame): 转换坐标系后的 GeoDataFrame。
    """
    # 读取 Shapefile 文件
    points =shapefile_path

    # 读取栅格文件的投影坐标系
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # 将 Shapefile 的坐标系转换为栅格图的投影坐标系
    points = points.to_crs(raster_crs)

    # 返回转换后的 GeoDataFrame
    return points
# def extract_raster_values(shapefile_path, raster_path, output_shapefile_path):
#     """
#     读取 Shapefile 文件，提取多波段遥感影像中对应点的栅格值，并将结果保存为新的 Shapefile 文件。
    
#     参数:
#         shapefile_path (str): 输入的 Shapefile 文件路径。
#         raster_path (str): 遥感影像文件路径。
#         output_shapefile_path (str): 输出的 Shapefile 文件路径。
#     """
#     # 读取 Shapefile 中的点
#     points = gpd.read_file(shapefile_path)

#     # 打开多波段遥感影像
#     with rasterio.open(raster_path) as src:
#         transform = src.transform
#         height, width = src.height, src.width
#         num_bands = src.count
#         band_data = src.read()  # 形状为 (num_bands, height, width)

#     # 将经纬度转换为栅格索引
#     x_coords, y_coords = points['geometry'].x, points['geometry'].y
#     row_indices, col_indices = rowcol(transform, x_coords, y_coords)

#     # 筛选有效点
#     valid_indices = [index for index, (row, col) in enumerate(zip(row_indices, col_indices)) 
#                      if 0 <= row < height and 0 <= col < width]
#     valid_points = points.loc[valid_indices].reset_index(drop=True)

#     # 提取每个波段的栅格值
#     band_values = {f'band_{i+1}': [] for i in range(num_bands)}
#     for band_idx in range(num_bands):
#         band_array = band_data[band_idx]
#         for row, col in zip(row_indices, col_indices):
#             if 0 <= row < height and 0 <= col < width:
#                 band_values[f'band_{band_idx+1}'].append(float(band_array[row, col]))
#             else:
#                 band_values[f'band_{band_idx+1}'].append(np.nan)

#     # 创建 DataFrame 并处理无效值
#     df = pd.DataFrame(band_values).replace(-3.4028230607370965e+38, np.nan).dropna()
#     valid_points = valid_points.assign(**{f'band_{i+1}': df[f'band_{i+1}'] for i in range(num_bands)})

#     # 保存为新的 Shapefile
#     valid_points.to_file(output_shapefile_path, driver='ESRI Shapefile')
#     print(f"新的 Shapefile 已保存到: {output_shapefile_path}")

def extract_raster_values(shapefile_path, raster_path):
    """
    读取 Shapefile 文件，提取多波段遥感影像中对应点的栅格值。
    
    参数:
        shapefile_path (str): 输入的 Shapefile 文件路径。
        raster_path (str): 遥感影像文件路径。
    
    返回:
        valid_points (GeoDataFrame): 包含提取的栅格值的 GeoDataFrame。
    """
    # 读取 Shapefile 中的点
    points = shapefile_path

    # 打开多波段遥感影像
    with rasterio.open(raster_path) as src:
        transform = src.transform
        height, width = src.height, src.width
        num_bands = src.count
        band_data = src.read()  # 形状为 (num_bands, height, width)

    # 将经纬度转换为栅格索引
    x_coords, y_coords = points['geometry'].x, points['geometry'].y
    row_indices, col_indices = rowcol(transform, x_coords, y_coords)

    # 筛选有效点
    valid_indices = [index for index, (row, col) in enumerate(zip(row_indices, col_indices)) 
                     if 0 <= row < height and 0 <= col < width]
    valid_points = points.loc[valid_indices].reset_index(drop=True)

    # 提取每个波段的栅格值
    band_values = {f'band_{i+1}': [] for i in range(num_bands)}
    for band_idx in range(num_bands):
        band_array = band_data[band_idx]
        for row, col in zip(row_indices, col_indices):
            if 0 <= row < height and 0 <= col < width:
                band_values[f'band_{band_idx+1}'].append(float(band_array[row, col]))
            else:
                band_values[f'band_{band_idx+1}'].append(np.nan)

    # 创建 DataFrame 并处理无效值
    df = pd.DataFrame(band_values).replace(-3.4028230607370965e+38, np.nan).dropna()
    valid_points = valid_points.assign(**{f'band_{i+1}': df[f'band_{i+1}'] for i in range(num_bands)})

    # 返回包含提取值的 GeoDataFrame
    return valid_points

# def shapefile_to_excel(shapefile_path, output_excel_path):
#     """
#     将 Shapefile 文件转换为 Excel 表格。
    
#     参数:
#         shapefile_path (str): Shapefile 文件路径（包括 .shp 文件）。
#         output_excel_path (str): 输出的 Excel 文件路径（包括 .xlsx 文件）。
#     """
#     # 读取 Shapefile 文件
#     gdf = shapefile_path
#     # 提取几何数据（如点的坐标）并添加到属性表中
#     if gdf.geometry.type.iloc[0] == 'Point':  # 如果是点数据
#         gdf['lon'] = gdf.geometry.x  # 提取经度
#         gdf['lat'] = gdf.geometry.y  # 提取纬度

#     # 将 GeoDataFrame 转换为普通的 DataFrame
#     df = pd.DataFrame(gdf.drop(columns='geometry'))

#     # 删除包含缺失值的行
#     df = df.dropna()

#     # 保存为 Excel 文件
#     df.to_excel(output_excel_path, index=False)
#     print(f"Excel 文件已保存到: {output_excel_path}")

import pandas as pd
from io import BytesIO

def shapefile_to_excel(gdf):
    """
    将 GeoDataFrame 转换为 Excel 表格，并返回 Excel 文件的二进制数据。
    
    参数:
        gdf (GeoDataFrame): 输入的 GeoDataFrame。
        
    返回:
        output_excel_bytes (BytesIO): Excel 文件的二进制数据。
    """
    # 提取几何数据（如点的坐标）并添加到属性表中
    if gdf.geometry.type.iloc[0] == 'Point':  # 如果是点数据
        gdf['lon'] = gdf.geometry.x  # 提取经度
        gdf['lat'] = gdf.geometry.y  # 提取纬度

    # 将 GeoDataFrame 转换为普通的 DataFrame
    df = pd.DataFrame(gdf.drop(columns='geometry'))

    # 删除包含缺失值的行
    df = df.dropna()

    # 将 DataFrame 保存为 Excel 文件的二进制数据
    output_excel_bytes = BytesIO()
    with pd.ExcelWriter(output_excel_bytes, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    
    # 将指针重置到文件开头
    output_excel_bytes.seek(0)
    
    return output_excel_bytes
################################################################################################################
def read_and_extract_dn_values(image_path, df):
    """
    读取多波段影像，根据 DataFrame 中的坐标位置提取 DN 值，并生成一个新的 Excel 表。
    如果点不在栅格图像中，则删除该点。DN 值保存为浮点型。

    参数:
        image_path (str): 多波段影像的文件路径。
        df (DataFrame): 包含坐标信息的 DataFrame。

    返回:
        temp_file_path (str): 临时文件的路径。
    """
    # 检查 image_path 是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"遥感影像文件未找到: {image_path}")

    # 检查 DataFrame 是否包含必要的列
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("DataFrame 中必须包含 'lat' 和 'lon' 列")

    # 打开多波段影像
    with rasterio.open(image_path) as src:
        # 获取波段数量和影像尺寸
        num_bands = src.count
        height, width = src.height, src.width  # 获取影像的高度和宽度

        # 初始化一个列表，用于存储需要保留的行索引
        valid_indices = []

        # 遍历 DataFrame 中的每一行，检查点是否在栅格图像中
        for index, row in df.iterrows():
            lat, lon = row['lat'], row['lon']

            # 将经纬度坐标转换为影像的行列号
            row, col = src.index(lon, lat)

            # 检查行列号是否在影像范围内
            if 0 <= row < height and 0 <= col < width:
                valid_indices.append(index)
            else:
                print(f"警告: 点 ({lat}, {lon}) 不在栅格图像中，已删除")

        # 只保留有效点
        df = df.loc[valid_indices].reset_index(drop=True)

        # 逐个波段读取并提取DN值
        for band_idx in range(1, num_bands + 1):  # 波段索引从 1 开始
            band_array = src.read(band_idx)  # 读取当前波段

            # 初始化一个列表来存储当前波段的DN值
            dn_values = []

            # 遍历有效点，提取对应坐标的DN值
            for index, row in df.iterrows():
                lat, lon = row['lat'], row['lon']

                # 将经纬度坐标转换为影像的行列号
                row, col = src.index(lon, lat)

                # 提取对应位置的DN值，并转换为浮点型
                dn_value = float(band_array[row, col])
                dn_values.append(dn_value)

            # 将当前波段的DN值添加到DataFrame中
            df[f'Band_{band_idx}_DN'] = dn_values

    # 创建一个临时文件
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
        temp_file_path = tmp_file.name  # 获取临时文件的路径

        # 将修改后的 DataFrame 保存到临时文件
        df.to_excel(temp_file_path, index=False)
        print(f"修改后的 Excel 文件已保存到临时文件: {temp_file_path}")

    # 返回临时文件路径
    return temp_file_path

################################################################################################################

# def read_and_process_bands(image_path):
#     """
#     读取多波段影像并为每个波段创建对应的二维 numpy 数组。

#     参数:
#         image_path (str): 多波段影像的文件路径。

#     返回:
#         band_arrays (list): 包含每个波段二维 numpy 数组的列表。
#         num_bands (int): 影像的波段数量。
#         metadata (dict): 影像的元数据。
#     """
#     # 检查文件是否存在
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"文件未找到: {image_path}")

#     # 打开多波段影像
#     with rasterio.open(image_path) as src:
#         # 获取波段数量和影像尺寸
#         num_bands = src.count
#         height = src.height
#         width = src.width
#         metadata = src.meta  # 获取元数据

#         # 创建一个空列表，用于存储每个波段的二维数组
#         band_arrays = []

#         # 逐个波段读取
#         for band_idx in range(1, num_bands + 1):  # 波段索引从 1 开始
#             band_array = src.read(band_idx)  # 读取当前波段

#             # 处理无效值（如负值）
#             band_array = np.where(band_array < 0, np.nan, band_array)  # 将负值替换为 NaN
#             mean_value = np.nanmean(band_array)  # 计算非 NaN 值的均值
#             band_array = np.where(np.isnan(band_array), mean_value, band_array)  # 用均值填充 NaN

#             band_arrays.append(band_array)  # 将处理后的二维数组添加到列表中

#     # 返回结果
#     return band_arrays, num_bands, metadata


def read_and_process_bands(image_path):
    """
    读取多波段影像并为每个波段创建对应的二维 numpy 数组，并添加坐标信息作为最后两个波段。

    参数:
        image_path (str): 多波段影像的文件路径。

    返回:
        band_arrays (list): 包含处理后的波段数组和坐标波段的列表。
        num_bands (int): 原始波段数量 + 2（坐标波段）。
        metadata (dict): 更新后的元数据。
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件未找到: {image_path}")

    with rasterio.open(image_path) as src:
        # 获取原始波段数量和元数据
        num_bands_original = src.count
        metadata = src.meta.copy()
        height, width = src.height, src.width

        # 生成坐标网格
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        transform = src.transform
        # 计算地理坐标
        xs = transform.a * cols + transform.b * rows + transform.c
        ys = transform.d * cols + transform.e * rows + transform.f

        # 确定数据类型（使用第一个波段处理后的类型）
        sample_band = src.read(1)
        # 处理后的波段可能转为float，这里统一为float32
        dtype = np.float32
        xs = xs.astype(dtype)
        ys = ys.astype(dtype)

        band_arrays = []

        # 处理原始波段
        for band_idx in range(1, num_bands_original + 1):
            band_array = src.read(band_idx)
            # 处理无效值
            band_array = np.where(band_array < 0, np.nan, band_array)
            mean_value = np.nanmean(band_array)
            band_array = np.where(np.isnan(band_array), mean_value, band_array)
            band_array = band_array.astype(dtype)
            band_arrays.append(band_array)

        # 添加坐标波段
        band_arrays.append(xs)
        band_arrays.append(ys)

        # 更新元数据
        metadata.update({
            'count': num_bands_original + 2,  # 新增两个波段
            'dtype': dtype
        })

    return band_arrays, num_bands_original + 2, metadata
# def read_and_process_bands(image_path):
#     """
#     读取多波段影像并为每个波段创建对应的二维 numpy 数组。

#     参数:
#         image_path (str): 多波段影像的文件路径。

#     返回:
#         band_arrays (list): 包含每个波段二维 numpy 数组的列表。
#         num_bands (int): 影像的波段数量。
#         metadata (dict): 影像的元数据。
#     """
#     # 检查文件是否存在
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"文件未找到: {image_path}")

#     # 打开多波段影像
#     with rasterio.open(image_path) as src:
#         # 获取波段数量和影像尺寸
#         num_bands = src.count
#         height = src.height
#         width = src.width
#         metadata = src.meta  # 获取元数据

#         # 创建一个空列表，用于存储每个波段的二维数组
#         band_arrays = []

#         # 逐个波段读取
#         for band_idx in range(1, num_bands + 1):  # 波段索引从 1 开始
#             band_array = src.read(band_idx)  # 读取当前波段

#             # 处理无效值（如负值）
#             band_array = np.where(band_array < 0, np.nan, band_array)  # 将负值替换为 NaN
#             mean_value = np.nanmean(band_array)  # 计算非 NaN 值的均值
#             band_array = np.where(np.isnan(band_array), mean_value, band_array)  # 用均值填充 NaN

#             band_arrays.append(band_array)  # 将处理后的二维数组添加到列表中

#     # 返回结果
#     return band_arrays, num_bands, metadata
#################################################################################################################
# def predict_flood_risk(model_path, X_array, reference_raster_path, output_file, height, width):
#     """
#     使用预训练模型预测洪水风险，并将结果保存为栅格文件。

#     参数:
#         model_path (str): 预训练模型的路径。
#         X_array (np.ndarray): 输入特征数据，形状为 (样本数, 特征数)。
#         reference_raster_path (str): 参考栅格文件的路径，用于获取空间信息。
#         output_file (str): 预测结果保存路径。
#         height (int): 输出栅格的高度。
#         width (int): 输出栅格的宽度。

#     返回:
#         Y_predict_proba (np.ndarray): 预测的洪水概率。
#         Y_predict (np.ndarray): 预测的洪水分类结果。
#     """
#     # 忽略警告
#     warnings.filterwarnings("ignore", category=UserWarning)

#     # 加载模型
#     # model = load(model_path)

#     # 预测洪水概率和分类结果
#     Y_predict_proba = model_path.predict_proba(X_array)
#     Y_predict = model_path.predict(X_array)

#     # 将洪水概率重塑为二维数组
#     flood_probabilities = Y_predict_proba[:, 1]  # 获取洪水类的概率
#     flood_probabilities_reshaped = flood_probabilities.reshape(height, width)

#     # 使用参考栅格的空间信息保存预测结果
#     with rasterio.open(reference_raster_path) as ref_src:
#         ref_meta = ref_src.meta.copy()
#         ref_meta.update({
#             "height": height,
#             "width": width,
#             "dtype": 'float32',  # 确保数据类型为 float32
#             "count": 1  # 确保只有一个波段
#         })

#     # 保存预测概率为栅格文件
#     with rasterio.open(output_file, 'w', **ref_meta) as dst:
#         dst.write(flood_probabilities_reshaped, 1)

#     print(f"预测概率已保存为栅格文件: {output_file}")

#     return Y_predict_proba, Y_predict,flood_probabilities_reshaped


import warnings
import numpy as np
import rasterio
from rasterio.io import MemoryFile

# def raster_to_bytes(raster_data):
#     """
#     将 rasterio.io.DatasetReader 对象转换为二进制数据。

#     参数:
#         raster_data (rasterio.io.DatasetReader): 栅格数据。

#     返回:
#         bytes: 栅格数据的二进制表示。
#     """
#     with MemoryFile() as memfile:
#         # 将栅格数据写入内存文件
#         with memfile.open(**raster_data.meta) as dst:
#             dst.write(raster_data.read(1), 1)  # 假设只有一个波段

#         # 将内存文件转换为二进制数据
#         return memfile.read()

# def predict_flood_risk(model_path, X_array, reference_raster_path, height, width):
#     """
#     使用预训练模型预测洪水风险，并将结果保存为栅格数据（内存中）。

#     参数:
#         model_path (str): 预训练模型的路径。
#         X_array (np.ndarray): 输入特征数据，形状为 (样本数, 特征数)。
#         reference_raster_path (str): 参考栅格文件的路径，用于获取空间信息。
#         height (int): 输出栅格的高度。
#         width (int): 输出栅格的宽度。

#     返回:
#         Y_predict_proba (np.ndarray): 预测的洪水概率。
#         Y_predict (np.ndarray): 预测的洪水分类结果。
#         flood_probabilities_reshaped (np.ndarray): 洪水概率的二维数组。
#         raster_data (rasterio.io.DatasetReader): 内存中的栅格数据。
#     """
#     # 忽略警告
#     warnings.filterwarnings("ignore", category=UserWarning)

#     # 加载模型
#     # model = load(model_path)

#     # 预测洪水概率和分类结果
#     Y_predict_proba = model_path.predict_proba(X_array)
#     Y_predict = model_path.predict(X_array)

#     # 将洪水概率重塑为二维数组
#     flood_probabilities = Y_predict_proba[:, 1]  # 获取洪水类的概率
#     flood_probabilities_reshaped = flood_probabilities.reshape(height, width)

#     # 使用参考栅格的空间信息创建栅格数据
#     with rasterio.open(reference_raster_path) as ref_src:
#         ref_meta = ref_src.meta.copy()
#         ref_meta.update({
#             "height": height,
#             "width": width,
#             "dtype": 'float32',  # 确保数据类型为 float32
#             "count": 1  # 确保只有一个波段
#         })

#     # 将栅格数据保存到内存中
#     with MemoryFile() as memfile:
#         with memfile.open(**ref_meta) as dst:
#             dst.write(flood_probabilities_reshaped.astype(np.float32), 1)  # 写入数据

#         # 将内存中的栅格数据作为变量返回
#         raster_data = memfile.open()

#         print("栅格数据已保存到内存中。")

#     return Y_predict_proba, Y_predict, flood_probabilities_reshaped, raster_data

def predict_flood_risk(model_path, X_array, reference_raster_path, height, width):
    """
    使用预训练模型预测洪水风险，并将结果保存为栅格数据（内存中）。

    参数:
        model_path (str): 预训练模型的路径。
        X_array (np.ndarray): 输入特征数据，形状为 (样本数, 特征数)。
        reference_raster_path (str): 参考栅格文件的路径，用于获取空间信息。
        height (int): 输出栅格的高度。
        width (int): 输出栅格的宽度。

    返回:
        Y_predict_proba (np.ndarray): 预测的洪水概率。
        Y_predict (np.ndarray): 预测的洪水分类结果。
        flood_probabilities_reshaped (np.ndarray): 洪水概率的二维数组。
        raster_binary_data (bytes): 栅格数据的二进制表示。
        raster_dataset (rasterio.DatasetReader): 内存中的栅格数据集对象。
    """
    # 忽略警告
    warnings.filterwarnings("ignore", category=UserWarning)

    # 加载模型
    model = model_path
    if hasattr(model, "n_features_in_"):
        expected_features = model.n_features_in_
        actual_features = X_array.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"特征数不匹配: 模型期望输入 {expected_features} 个特征，但实际输入了 {actual_features} 个特征。"
                "请检查输入数据的特征数是否与模型训练时的特征数一致。"
            )
    # 预测洪水概率和分类结果
    Y_predict_proba = model.predict_proba(X_array)
    Y_predict = model.predict(X_array)

    # 将洪水概率重塑为二维数组
    flood_probabilities = Y_predict_proba[:, 1]  # 获取洪水类的概率
    flood_probabilities_reshaped = flood_probabilities.reshape(height, width)

    # 使用参考栅格的空间信息创建栅格数据
    with rasterio.open(reference_raster_path) as ref_src:
        ref_meta = ref_src.meta.copy()
        ref_meta.update({
            "height": height,
            "width": width,
            "dtype": 'float32',  # 确保数据类型为 float32
            "count": 1  # 确保只有一个波段
        })

    # 将栅格数据保存到内存中并返回二进制数据
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**ref_meta) as dst:
            dst.write(flood_probabilities_reshaped.astype(np.float32), 1)  # 写入数据
        raster_binary_data = memfile.read()

    print("栅格数据已保存到内存中。")

    # 返回栅格数据集对象
    with rasterio.MemoryFile(raster_binary_data) as memfile:
        raster_dataset = memfile.open()

    return Y_predict_proba, Y_predict, flood_probabilities_reshaped, raster_binary_data, raster_dataset



            

