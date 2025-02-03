import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_folder(root_dir):
    """
    遍历文件夹，提取所有符合条件的影像路径
    返回结构：{ (region_code, year): [影像路径列表] }
    """
    data_dict = {}
    for dirpath, _, _ in os.walk(root_dir):
        # 检查是否有影像文件（假设为tif格式）
        if len(glob.glob(os.path.join(dirpath, '*.tif'))) > 0:
            # 解析路径中的年份和地区
            parts = dirpath.split(os.sep)
            if len(parts) >= 3:  # 确保路径深度足够
                year = parts[-2]  # 年份是倒数第二部分
                region = parts[-1]  # 地区是最后一部分
                data_dict[(region, year)] = sorted(
                    glob.glob(os.path.join(dirpath, '*.tif')),
                    key=lambda x: int(os.path.splitext(x)[0].split('-')[-1])  # 假设文件名包含时间步编号
                )
    return data_dict


def extract_features(image_path):
    """从单张影像中提取特征（这里使用波段均值作为示例）"""
    with rasterio.open(image_path) as src:
        img = src.read()  # 读取所有波段 (7, H, W)
    # 计算每个波段的均值（可根据需求修改为其他统计量）
    return [np.nanmean(band) for band in img]


def create_dataframe(data_dict):
    """创建包含时空特征的数据框架"""
    rows = []
    for (region, year), img_paths in tqdm(data_dict.items()):
        # 检查时间步数量是否为32
        if len(img_paths) != 32:
            print(f"Skipping {region}_{year}: 时间步数量不足32")
            continue

        # 提取所有时间步的特征
        time_series = [[] for _ in range(7)]  # 7个波段的时序数据
        for path in img_paths:
            features = extract_features(path)
            for i in range(7):
                time_series[i].append(features[i])

        # 创建行数据
        row = {'Region': region, 'Year': year}
        for i in range(7):
            row[f'Band_{i + 1}'] = time_series[i]
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    input_dir = r"E:\25holiday\Dataset\Finally\lzw_mod09a1"  # 修改为你的根目录
    output_file = "output_features.xlsx"

    # 处理数据
    data_dict = process_folder(input_dir)
    df = create_dataframe(data_dict)

    # 保存到Excel（注意：需要安装openpyxl）
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"数据已保存至 {output_file}")