import os
from datetime import datetime
from config.config import option as opt
import numpy as np
import rasterio
from torch.utils.data import Dataset

import h5py
import pandas as pd

from utils.get_history import get_yield_history


class ImageSequenceDataset(Dataset):
    def __init__(self, folders_09a1, folders_11a2, folders_fldas,
                 hdf5_09a1='data/data_09a1.h5',
                 hdf5_11a2='data/data_11a2.h5',
                 hdf5_fldas='data/data_fldas.h5',
                 min_val=None, max_val=None, is_train=True):  # 添加 use_h5 参数
        self.labels = []
        self.path = []
        self.area = []
        self.history_data = []
        max_len = 0
        min_len = 0

        df = pd.read_csv('data/area_all.csv')

        # 遍历每个文件夹
        for i, folder in enumerate(folders_09a1):
            # 获取文件夹中的所有图片文件
            files = os.listdir(folder)
            self.labels.append(self._read_label_and_convert(folder))
            self.path.append(folder)
            year, code = folder.split(opt.split_str)[-2:]
            self.history_data.append(get_yield_history(str(year), str(code)))
            if not is_train:
                filtered_df = df[(df['Year'] == int(year)) & (df['Code'] == int(code))]['Area']
                self.area.append(float(filtered_df))
            else:
                self.area.append(0)

            length = len(files)
            if max_len == 0:
                max_len = length
            if min_len == 0:
                min_len = length
            if length > max_len:
                max_len = length
            if length < min_len:
                min_len = length

        self.labels = np.array(self.labels)
        # 计算原始标签的最小值和最大值
        raw_min_val = np.min(self.labels)
        raw_max_val = np.max(self.labels)

        if opt.label_nor:
            # 对标签进行归一化
            if min_val is not None and max_val is not None:
                self.labels = opt.norm_ratio * (self.labels - min_val) / (max_val - min_val)
            else:
                self.labels = opt.norm_ratio * (self.labels - raw_min_val) / (raw_max_val - raw_min_val)

        # 返回原始标签的最小值和最大值
        self.raw_min_val = raw_min_val
        self.raw_max_val = raw_max_val
        self.labels = self.labels.astype(np.float32)
        self.history_data = self.pad_sequences_to_length(self.history_data)
        self.history_data = np.array(self.history_data)
        self.history_data = self.history_data.astype(np.float32)

        # 构建或加载数据
        if not self.check_file_exists(hdf5_09a1):
            self._build_or_load_h5(hdf5_09a1, folders_09a1, 7, opt.img_shape)
        if not self.check_file_exists(hdf5_11a2):
            self._build_or_load_h5(hdf5_11a2, folders_11a2, 2, opt.img_shape)
        if not self.check_file_exists(hdf5_fldas):
            self._build_or_load_h5(hdf5_fldas, folders_fldas, 8, opt.fldas_shape)
        print(f'load to memory===============')
        self.data_09a1 = self._load_data_to_memory(folders_09a1, 7, hdf5_09a1, opt.img_shape)
        self.data_11a2 = self._load_data_to_memory(folders_11a2, 2, hdf5_11a2, opt.img_shape)
        self.data_fldas = self._load_data_to_memory(folders_fldas, 8, hdf5_fldas, opt.fldas_shape)

        print(f'max length:{max_len},min length:{min_len}')
        print(f'max label {max(self.labels)}, min label {min(self.labels)}')
        print(
            f'Dataset Finished! data_09a1 length:{len(self.data_09a1) if isinstance(self.data_09a1, list) else len(self.data_09a1.keys())}, data_11a2 length:{len(self.data_11a2) if isinstance(self.data_11a2, list) else len(self.data_11a2.keys())} labels length:{len(self.labels)} area length: {len(self.area)}')
        if len(self.data_09a1) != len(self.data_11a2) or len(self.data_09a1) != len(self.labels) or len(
                self.data_09a1) != len(self.area):
            raise ValueError('data length error')

    def _read_label_and_convert(self, folder_path):  # 示例标签读取和转换函数
        file_path = os.path.join(folder_path, 'label', 'label.txt')

        # 打开并读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 检查第二行的内容
        if lines[1].strip() == '吨':
            value = float(lines[0].strip()) / 10000
        elif lines[1].strip() == '万吨':
            value = float(lines[0].strip())
        else:
            raise ValueError('产量数据出现未知的单位！')
        return value

    def _build_or_load_h5(self, hdf5_file, folders, channel, shape):
        # 检查h5文件是否存在
        if os.path.exists(hdf5_file):
            data = h5py.File(hdf5_file, 'r')
        else:
            print(f'No H5 file {hdf5_file}, create h5 file===============================')
            data = self._build_h5(hdf5_file, folders, channel, shape)
        return data

    def _build_h5(self, hdf5_file, folders, channel, shape):
        with h5py.File(hdf5_file, 'w') as data:
            for i, folder in enumerate(folders):
                files = os.listdir(folder)
                files = [f for f in files if f.endswith('.tif')]
                length = len(files)
                files = sorted(files,
                               key=lambda x: datetime.strptime(os.path.splitext(os.path.splitext(x)[0])[0], '%Y-%m-%d'))
                images = data.create_dataset(f'dataset_{i}', (length, channel, shape, shape),
                                             dtype='float32')
                for j, file in enumerate(files):
                    with rasterio.open(os.path.join(folder, file)) as src:
                        img = src.read()
                    img = self._preprocess_image(img, shape)
                    images[j] = img
        return h5py.File(hdf5_file, 'r')

    def _load_data_to_memory(self, folders, channel, hdf5_file, shape):
        if os.path.exists(hdf5_file):  # 判断 h5 文件是否存在
            print(f"Loading data from existing H5 file: {hdf5_file}")
            with h5py.File(hdf5_file, 'r') as hf:
                data = []
                for i in range(len(folders)):
                    data.append(hf[f'dataset_{i}'][:])  # 直接从 h5 文件加载数据
            return data
        else:
            print(f"H5 file {hdf5_file} not found. Loading data from folders.")
            data = []
            for i, folder in enumerate(folders):
                files = os.listdir(folder)
                files = [f for f in files if f.endswith('.tif')]
                length = len(files)
                files = sorted(files,
                               key=lambda x: datetime.strptime(os.path.splitext(os.path.splitext(x)[0])[0], '%Y-%m-%d'))
                images = np.empty((length, channel, shape, shape), dtype=np.float32)
                for j, file in enumerate(files):
                    with rasterio.open(os.path.join(folder, file)) as src:
                        img = src.read()
                    img = self._preprocess_image(img, shape)
                    images[j] = img
                data.append(images)
            return data

    def _preprocess_image(self, img, shape):
        """预处理图像，包括裁剪和填充。

        Args:
            img (numpy.ndarray): 输入图像，形状为 (C, H, W)。

        Returns:
            numpy.ndarray: 预处理后的图像，形状为 (C, opt.img_shape, opt.img_shape)。
        """
        if img.shape[1] > shape or img.shape[2] > shape:
            raise ValueError(f'image shape error! {img.shape[1]},{img.shape[2]}')
        if img.shape[1] < shape:
            # 如果图像宽度小于目标宽度，则在右侧进行零填充
            padding = np.zeros((img.shape[0], shape - img.shape[1], img.shape[2]), dtype=img.dtype)
            img = np.concatenate((img, padding), axis=1)
        if img.shape[2] < shape:
            # 如果图像高度小于目标高度，则在底部进行零填充
            padding = np.zeros((img.shape[0], img.shape[1], shape - img.shape[2]), dtype=img.dtype)
            img = np.concatenate((img, padding), axis=2)
        return img

    def check_file_exists(self, file_path):
        if os.path.exists(file_path):
            return True
        return False

    def pad_sequences_to_length(self, sequences, desired_length=25):
        """
        对列表中的每个元素进行长度检查，并将长度不足20的元素填充为 NaN。

        参数:
        - sequences: 包含多个序列的列表，每个序列是一个 numpy 数组
        - desired_length: 目标长度，默认为 20

        返回:
        - 填充后的列表，每个元素长度均为 desired_length
        """
        padded_sequences = []
        for seq in sequences:
            if len(seq) < desired_length:
                # 计算需要填充的 NaN 数量
                pad_width = desired_length - len(seq)
                # 使用 np.pad 进行填充
                padded_seq = np.pad(seq, (0, pad_width), mode='constant', constant_values=np.nan)
            else:
                # 如果长度已经大于或等于目标长度，则不进行填充
                padded_seq = seq
            padded_sequences.append(padded_seq)
        return padded_sequences

    def __len__(self):
        return len(self.data_09a1)

    def __getitem__(self, idx):
        if isinstance(self.data_09a1, h5py.File):  # 判断数据类型
            data_09a1 = self.data_09a1[f'dataset_{idx}'][:]
            data_11a2 = self.data_11a2[f'dataset_{idx}'][:]
            data_fldas = self.data_fldas[f'dataset_{idx}'][:]
        else:
            data_09a1 = self.data_09a1[idx]
            data_11a2 = self.data_11a2[idx]
            data_fldas = self.data_fldas[idx]

        return data_09a1, data_11a2, data_fldas, self.labels[idx], self.path[idx], self.area[idx], self.history_data[idx]


if __name__ == '__main__':
    # 数据集文件夹列表
    val_list = []

    # 打开并读取txt文件
    with open('data/val.txt', 'r') as f:
        for line in f:
            # 去除每行的换行符，并添加到列表中
            val_list.append(line.strip())
    val_dataset = ImageSequenceDataset(val_list, hdf5_file='data/val_data.h5')
