import os
import re
import numpy as np
import rasterio
from datetime import datetime
from typing import List


class SatelliteTimeSeriesDataset:
    def __init__(self, folder_a_list: List[str], folder_b_list: List[str]):
        # 验证输入参数有效性
        self._validate_inputs(folder_a_list, folder_b_list)

        # 存储原始参数
        self.folder_a_list = folder_a_list
        self.folder_b_list = folder_b_list

        # 初始化数据结构
        self.feature_matrix = []
        self.labels = []
        self.group_indices = []  # 记录每个组的样本范围

        # 处理每组数据
        self._process_groups()

        # 转换为numpy数组
        self.feature_matrix = np.concatenate(self.feature_matrix, axis=0).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def _validate_inputs(self, folder_a_list, folder_b_list):
        """验证输入参数有效性"""
        if not isinstance(folder_a_list, list) or not isinstance(folder_b_list, list):
            raise TypeError("folder_a和folder_b必须是列表类型")

        if len(folder_a_list) != len(folder_b_list):
            raise ValueError("folder_a和folder_b列表长度必须相同")

        for a, b in zip(folder_a_list, folder_b_list):
            if not os.path.isdir(a):
                raise NotADirectoryError(f"folder_a路径不存在: {a}")
            if not os.path.isdir(b):
                raise NotADirectoryError(f"folder_b路径不存在: {b}")

    def _load_group_label(self, folder_a: str) -> float:
        """加载单个数据组的标签"""
        label_dir = os.path.join(folder_a, 'label')
        label_path = os.path.join(label_dir, 'label.txt')

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) < 2:
                raise ValueError(f"标签文件格式错误: {label_path}")

            value_str, unit = lines[0], lines[1]

        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"无效的数值格式: {value_str}")

        # 单位统一转换为吨
        unit_conversion = {
            '吨': 1,
            '万吨': 10000
        }

        if unit not in unit_conversion:
            raise ValueError(f"不支持的单位类型: {unit}")

        return value * unit_conversion[unit]

    def _process_single_group(self, folder_a: str, folder_b: str) -> np.ndarray:
        """处理单个数据组"""
        # 加载标签
        label = self._load_group_label(folder_a)

        # 获取并验证文件列表
        files_a = self._get_sorted_files(folder_a)
        files_b = self._get_sorted_files(folder_b)

        if len(files_a) != len(files_b):
            raise ValueError(
                f"数据组文件数量不匹配\n"
                f"Folder A ({len(files_a)}): {files_a[:3]}...\n"
                f"Folder B ({len(files_b)}): {files_b[:3]}..."
            )

        # 预处理特征
        group_features = []
        for a_file, b_file in zip(files_a, files_b):
            a_path = os.path.join(folder_a, a_file)
            b_path = os.path.join(folder_b, b_file)

            features = np.concatenate([
                self._process_image(a_path),
                self._process_image(b_path)
            ])
            group_features.append(features)

        return np.array(group_features), label

    def _process_groups(self):
        """处理所有数据组"""
        total_samples = 0
        for idx, (a, b) in enumerate(zip(self.folder_a_list, self.folder_b_list)):
            # 处理单个数据组
            group_features, group_label = self._process_single_group(a, b)

            # 记录分组信息
            num_samples = group_features.shape[0]
            self.group_indices.append((total_samples, total_samples + num_samples))
            total_samples += num_samples

            # 存储数据
            self.feature_matrix.append(group_features)
            self.labels.extend([group_label] * num_samples)  # 每个样本对应同一标签

    def _extract_date(self, filename: str) -> datetime:
        """从文件名提取日期"""
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
        if not date_match:
            raise ValueError(f"文件名中未找到有效日期: {filename}")

        try:
            return datetime.strptime(date_match.group(), "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"无效的日期格式: {date_match.group()}")

    def _get_sorted_files(self, folder: str) -> List[str]:
        """获取排序后的文件列表"""
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.img'))]
        return sorted(files, key=lambda x: self._extract_date(x))

    def _process_image(self, file_path: str) -> np.ndarray:
        """处理单张影像"""
        with rasterio.open(file_path) as src:
            data = src.read()  # (bands, height, width)

        features = []
        for band in data:
            valid_pixels = band[band != 0]
            features.append(valid_pixels.mean() if valid_pixels.size > 0 else 0.0)

        return np.array(features)

    def __len__(self) -> int:
        """获取数据集总长度"""
        return self.feature_matrix.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """获取单个样本"""
        return self.feature_matrix[idx], self.labels[idx]

    @property
    def shape(self) -> tuple:
        """数据集形状 (样本数, 特征数)"""
        return self.feature_matrix.shape

    def get_group_info(self, idx: int) -> dict:
        """获取指定数据组的信息"""
        start, end = self.group_indices[idx]
        return {
            'folder_a': self.folder_a_list[idx],
            'folder_b': self.folder_b_list[idx],
            'samples_range': (start, end),
            'label': self.labels[start]  # 同一组的标签相同
        }


# 示例用法
if __name__ == "__main__":
    # 假设有两个数据组
    dataset = SatelliteTimeSeriesDataset(
        folder_a_list=[
            r"E:\25holiday\Dataset\Finally\lzw_mod09a1\Gansu\2002\620502",
            r"E:\25holiday\Dataset\Finally\lzw_mod09a1\Gansu\2002\620503"
        ],
        folder_b_list=[
            r"E:\25holiday\Dataset\Finally\lzw_mod11a2\Gansu\2002\620502",
            r"E:\25holiday\Dataset\Finally\lzw_mod11a2\Gansu\2002\620503"
        ]
    )

    print(f"总样本数: {len(dataset)}")
    print(f"特征维度: {dataset.shape[1]}")

    # 获取第一个样本
    sample, label = dataset[0]
    print("\n第一个样本:")
    print(f"特征向量: {sample[:5]}... (长度: {len(sample)})")
    print(f"对应标签: {label} 吨")

    # 获取分组信息
    group_info = dataset.get_group_info(0)
    print("\n第一组信息:")
    print(f"数据路径A: {group_info['folder_a']}")
    print(f"数据范围: {group_info['samples_range'][0]}~{group_info['samples_range'][1]}")