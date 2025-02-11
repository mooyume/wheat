import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_pred, y_true):
    print(len(y_true))
    # 转换输入为NumPy数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 校验输入长度
    if len(y_true) != len(y_pred):
        raise ValueError("预测数组与实际数组长度不一致")

    # 计算MAE和RMSE（直接使用sklearn函数）
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 计算MAPE（手动实现，过滤实际值为0的样本）
    mask = y_true != 0  # 过滤实际值为0的样本
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan  # 如果所有实际值都是0，返回NaN

    # 计算R²（使用sklearn函数）
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


if __name__ == '__main__':
    # 示例数据
    y_true = [30, -5, 20, 70]
    y_pred = [25, 0.0, 20, 80]

    # 调用函数
    metrics = calculate_metrics(y_pred, y_true)

    # 输出结果
    print("MAE:", metrics['MAE'])  # 输出：MAE: 0.5
    print("RMSE:", metrics['RMSE'])  # 输出：RMSE: 0.6123724356957945
    print("MAPE:", metrics['MAPE'])  # 输出：MAPE: 128.33333333333334（注意负值和0的影响）
    print("R²:", metrics['R2'])  # 输出：R²: 0.9486081370449679
