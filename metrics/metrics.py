import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_util import remove_item

"""
RMSE   kg*hm^-2
MAPE   %
R^2

MAE  kg*hm^-2
"""


# 平均绝对百分比误差  越小越好
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten())) * 100


def calculate_mape_2(y_true, y_pred, sowing_area):
    y_true_kg = y_true.flatten() * 10 ** 7
    y_pred_kg = y_pred.flatten() * 10 ** 7

    # 计算每公顷的千克数
    y_true_kg_per_hm2 = y_true_kg / sowing_area.flatten()
    y_pred_kg_per_hm2 = y_pred_kg / sowing_area.flatten()
    return np.mean(np.abs((y_true_kg_per_hm2 - y_pred_kg_per_hm2) / y_true.flatten())) * 100


# 均方根误差 越小越好
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))


# 平均绝对误差  越小越好
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())


# 均方根误差 越小越好
def calculate_rmse_2(y_true, y_pred, sowing_area):
    # 将真实值和预测值从万吨转换为千克
    y_true_kg = y_true.flatten() * 10 ** 7
    y_pred_kg = y_pred.flatten() * 10 ** 7

    # 计算每公顷的千克数
    y_true_kg_per_hm2 = y_true_kg / sowing_area.flatten()
    y_pred_kg_per_hm2 = y_pred_kg / sowing_area.flatten()

    return np.sqrt(mean_squared_error(y_true_kg_per_hm2, y_pred_kg_per_hm2))


# 平均绝对误差  越小越好
def calculate_mae_2(y_true, y_pred, sowing_area):
    # 将真实值和预测值从万吨转换为千克
    y_true_kg = y_true.flatten() * 10 ** 7
    y_pred_kg = y_pred.flatten() * 10 ** 7

    # 计算每公顷的千克数
    y_true_kg_per_hm2 = y_true_kg / sowing_area.flatten()
    y_pred_kg_per_hm2 = y_pred_kg / sowing_area.flatten()

    return mean_absolute_error(y_true_kg_per_hm2, y_pred_kg_per_hm2)


# 决定系数   越接近于1越好
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def calculate_r2_2(y_true, y_pred, sowing_area):
    # 将真实值和预测值从万吨转换为千克
    y_true_kg = y_true.flatten() * 10 ** 7
    y_pred_kg = y_pred.flatten() * 10 ** 7

    # 计算每公顷的千克数
    y_true_kg_per_hm2 = y_true_kg / sowing_area.flatten()
    y_pred_kg_per_hm2 = y_pred_kg / sowing_area.flatten()
    return r2_score(y_true_kg_per_hm2, y_pred_kg_per_hm2)


def metric(pred, label, sowing_area, logger, path):
    print(f'array len: {len(pred)}')
    if np.any(sowing_area == 0):
        raise ValueError('area can not includes zero!!')
    real = (label * 10 ** 7) / sowing_area
    pred = (pred * 10 ** 7) / sowing_area

    # pred, real, area, _, k = remove_item(pred, real, sowing_area, path, value=1000)
    rmse, mae, mape, r2, relative_error = calculate_metrics(pred, real)
    logger.info(f'rmse:{rmse}, mae:{mae}, mape:{mape}, r2:{r2}')
    return mape, rmse, mae, r2, relative_error, 0


def calculate_metrics(pred, real):
    # 将列表转换为 numpy 数组
    array1 = np.array(pred)
    array2 = np.array(real)

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(array1, array2))

    # 计算 MAE
    mae = mean_absolute_error(array1, array2)

    # 计算 MAPE
    mape = np.mean(np.abs((array1 - array2) / array1)) * 100

    # 计算 R²
    r2 = r2_score(array1, array2)

    # 计算相对误差
    relative_error = np.abs((array1 - array2) / array2)
    return rmse, mae, mape, r2, relative_error

# if __name__ == '__main__':
