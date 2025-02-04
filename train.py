import os

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import option as opt
from dataset.dataset import ImageSequenceDataset
from log import setup_logger
from metrics.metrics import metric
from utils.data_util import append_string_to_file
from utils.model_util import build_model


# 定义Early Stopping类
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# 定义 rmse_loss 函数
def rmse_loss(input, target):
    return torch.sqrt(torch.mean((input - target) ** 2))


# 定义 loss_function 函数
def loss_function():
    loss_functions = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'rmse': rmse_loss,
        'huber': nn.HuberLoss()
    }

    return loss_functions.get(opt.loss_f, rmse_loss)


def process_file(file_path, old_str, new_str, fldas):
    list_old = []
    list_new = []
    list_fldas = []

    with open(file_path, 'r') as f:
        for line in f:
            # 去除每行的换行符，并添加到列表中
            list_old.append(line.strip())
            line = line.replace(old_str, new_str)
            list_new.append(line.strip())

            fldas_line = line.strip().replace(new_str, fldas)
            fldas_line = os.path.join(fldas_line, 'matched_temporal')
            list_fldas.append(fldas_line.strip())

    return list_old, list_new, list_fldas


def train():
    file_path = f'./{opt.save_path}/{opt.save_name}/'
    # 获取文件的目录路径
    directory = os.path.dirname(file_path)
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = setup_logger(file_path)
    logger.info(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'current device: {device}')

    model = build_model(logger, is_train=True)
    model.to(device)

    # 如果你想在每个训练周期开始时打乱数据，可以将此设置为 True
    shuffle = True

    # 数据集文件夹列表
    train_09a1, train_11a2, train_fldas = process_file(opt.train_file, 'mod09a1', 'mod11a2', 'fldas')
    val_09a1, val_11a2, val_fldas = process_file(opt.val_file, 'mod09a1', 'mod11a2', 'fldas')

    logger.info('Begin build dataset========================================')
    train_dataset = ImageSequenceDataset(train_09a1, train_11a2, train_fldas, hdf5_09a1=opt.train_h5_09a1,
                                         hdf5_11a2=opt.train_h5_11a2, hdf5_fldas=opt.train_h5_fldas)
    min_val, max_val = train_dataset.raw_min_val, train_dataset.raw_max_val
    val_dataset = ImageSequenceDataset(val_09a1, val_11a2, val_fldas, hdf5_09a1=opt.val_h5_09a1,
                                       hdf5_11a2=opt.val_h5_11a2, hdf5_fldas=opt.val_h5_fldas, min_val=min_val, max_val=max_val, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=shuffle)

    criterion = loss_function()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.decay_gamma,
                                                           patience=opt.n_lr_decay)

    early_stopping = EarlyStopping(patience=15, verbose=True)
    loss_values = []
    val_loss_values = []
    for epoch in range(opt.n_epochs):
        total_loss = 0.0
        # 打印当前的学习率
        for param_group in optimizer.param_groups:
            logger.info(f'Epoch: {epoch + 1}, Learning rate: {param_group["lr"]}')

        for i, (x, y, fldas, labels, path, _, history_data) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            fldas = y.to(device)
            history_data = y.to(device)
            labels = labels.view(-1, 1).to(device)
            outputs = model(x, y, fldas, history_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        total_loss /= len(train_dataloader)
        logger.info(f'Epoch: {epoch + 1}, Total Loss: {total_loss}')
        loss_values.append(total_loss)

        model.eval()
        val_loss = 0.0
        real_metric = numpy.array([])
        pred_metric = numpy.array([])
        area_metric = numpy.array([])
        path_metric = numpy.array([])
        with torch.no_grad():  # 在验证过程中不需要计算梯度
            for x, y, fldas, labels, path, area, history_data in val_dataloader:  # 假设val_dataloader是你的验证数据加载器
                x = x.to(device)
                y = y.to(device)
                labels = labels.view(-1, 1).to(device)
                outputs = model(x, y, fldas, history_data)
                val_loss += criterion(outputs, labels).item()  # 累加每个批次的损失
                if opt.label_nor:
                    real_outputs = (outputs / opt.norm_ratio) * (max_val - min_val) + min_val
                    real_labels = (labels / opt.norm_ratio) * (max_val - min_val) + min_val
                else:
                    real_outputs = outputs
                    real_labels = labels

                real_metric = numpy.append(real_metric, real_labels.detach().cpu().numpy().flatten())
                pred_metric = numpy.append(pred_metric, real_outputs.detach().cpu().numpy().flatten())
                area_metric = numpy.append(area_metric, area.detach().cpu().numpy().flatten())
                path_metric = numpy.append(path_metric, numpy.array(path))

        model.train()
        val_loss /= len(val_dataloader)
        mape, rmse, mae, r2, _, k = metric(pred_metric, real_metric, area_metric, logger, path_metric)
        text = f'remove:{k}, Epoch [{epoch + 1}/{opt.n_epochs}], Validation Loss: {val_loss}, MAPE: {mape}, RMSE: {rmse}, MAE: {mae}, R2: {r2}'
        logger.info(text)
        append_string_to_file(text,
                              os.path.join(file_path, 'metrics.txt'))
        val_loss_values.append(val_loss)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if (epoch + 1) % opt.plt_epochs == 0:
            plt.plot(loss_values, label='Train Loss')
            plt.plot(val_loss_values, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{file_path}/loss_{epoch + 1}.png')
            plt.close()
        if early_stopping.early_stop:
            logger.info(f'Early stopping at epoch {epoch + 1}')
            break
        if r2 > 0.7 or (epoch + 1) % opt.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(file_path, f'{opt.save_name}_{epoch + 1}.pth'))


if __name__ == '__main__':
    train()
