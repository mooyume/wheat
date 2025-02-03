import glob
import re

import torch

from config.config import option as opt
from model.Kansformer import Kansformer
from model.cnn3_pred import YieldPredictionModel
from model.cnn_transformer import Kansformer_lstm


def build_model(logger, is_train=False, is_test=False, model_name=None):
    logger.info(f'build model: {opt.model_name}====================')
    if opt.model_name == 'kan_two_branch':
        model = Kansformer(opt.x_channel, opt.y_channel)
    elif opt.model_name == '3d':
        model = YieldPredictionModel(time_steps=32, in_channels=(int(opt.x_channel) + int(opt.y_channel)))
    elif opt.model_name == '2cnn_lstm_kan':
        model = Kansformer_lstm(9, 8)
    else:
        raise ValueError('model name error, please check model name!!')
    logger.info(model)
    # 获取所有的模型文件
    model_files = glob.glob(f'{opt.save_path}/{opt.save_name}/{opt.save_name}_*.pth')

    if is_train:
        save_config_to_file(opt, f'{opt.save_path}/{opt.save_name}/config.txt')

    if model_name:
        logger.info(f"加载模型：{model_name}")
        # 加载模型参数
        model.load_state_dict(torch.load(model_name))
    # 如果有模型文件
    elif model_files:
        # 找到epoch最大的模型文件
        latest_model_file = max(model_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
        logger.info(f"加载模型：{latest_model_file}")
        # 加载模型参数
        model.load_state_dict(torch.load(latest_model_file))
    else:
        if is_test:
            raise ValueError('无法获取到参数模型！')
        logger.info("没有找到已保存的模型参数")
    # model = model.double()
    logger.info(f'Model parameter: {count_parameters(model)}M')
    return model


def save_config_to_file(option, file_path):
    try:
        # 写入配置参数到文件
        with open(file_path, 'w', encoding='utf-8') as file:
            for key, value in vars(option).items():
                file.write(f"{key}: {value}\n")
        print(f"配置参数已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"保存配置参数时发生错误: {e}")


def count_parameters(model):
    """
    计算并返回模型的总参数量（单位：百万）
    """
    total_params = sum(param.numel() for param in model.parameters())
    total_params_in_millions = total_params / 1_000_000
    return total_params_in_millions
