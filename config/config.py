import argparse
import sys

if sys.platform.startswith('linux'):
    default_path = '/root/autodl-tmp/'
    split_str = '/'
    file_paths = [r"/root/wheat/data/his_data/gansu-label.xlsx",
                  r"/root/wheat/data/his_data/henan-label.xlsx",
                  r"/root/wheat/data/his_data/shanxi-label.xlsx"]
elif sys.platform == 'win32':
    default_path = 'data/'
    split_str = '\\'
    file_paths = [r"E:\25holiday\data\label\gansu-label.xlsx",
                  r"E:\25holiday\data\label\henan-label.xlsx",
                  r"E:\25holiday\data\label\shanxi-label.xlsx"]
else:
    default_path = 'data/'
    split_str = '\\'
    file_paths = [r"E:\25holiday\data\label\gansu-label.xlsx",
                  r"E:\25holiday\data\label\henan-label.xlsx",
                  r"E:\25holiday\data\label\shanxi-label.xlsx"]

parser = argparse.ArgumentParser()
parser.add_argument('--x_channel', type=int, default=7, help='number of image channel')
parser.add_argument('--y_channel', type=int, default=2, help='number of image channel')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--train_file', type=str, default='data/train.txt', help='The file of train')
parser.add_argument('--val_file', type=str, default='data/val.txt', help='The file of val')
parser.add_argument('--test_file', type=str, default='data/val.txt', help='The file of test')

parser.add_argument('--train_h5_09a1', type=str, default=default_path + '/train_09a1.h5',
                    help='The h5 file of train')
parser.add_argument('--train_h5_fldas', type=str, default=default_path + '/train_fldas.h5',
                    help='The h5 file of train')

parser.add_argument('--val_h5_09a1', type=str, default=default_path + '/val_09a1.h5', help='The h5 file of val')
parser.add_argument('--val_h5_fldas', type=str, default=default_path + '/val_fldas.h5', help='The h5 file of val')

parser.add_argument('--test_h5_09a1', type=str, default=default_path + '/test_09a1.h5', help='The h5 file of test')
parser.add_argument('--test_h5_fldas', type=str, default=default_path + '/test_fldas.h5', help='The h5 file of test')

parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
parser.add_argument('--save_epochs', type=int, default=10, help='number of save model and plot loss image')
parser.add_argument('--plt_epochs', type=int, default=10, help='number of save model and plot loss image')
parser.add_argument('--loss_step', type=int, default=10, help='step of logging loss')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--n_lr_decay', type=int, default=10, help='step of lr decay')
parser.add_argument('--decay_gamma', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--log_path', type=str, default='log', help='log file path')
parser.add_argument('--save_path', type=str, default='pt', help='model save path')
parser.add_argument('--model_name', type=str, default='2cnn_encoder_lstm',
                    help='save model name [kan_two_branch, 3d, 2cnn_encoder_lstm]')
parser.add_argument('--save_name', type=str, default='two_encoder_cbam_cross_gate_guanzhong', help='save file name')
parser.add_argument('--accumulation_steps', type=int, default=1,
                    help='accumulation steps real batch size is accumulation_steps*batch_size')
parser.add_argument('--norm_ratio', type=int, default=10, help='norm ratio')

parser.add_argument('--img_shape', type=int, default=134, help='img_shape')
parser.add_argument('--fldas_shape', type=int, default=16, help='img_shape')
parser.add_argument('--time_att', type=bool, default=True, help='step pooling')
parser.add_argument('--label_nor', type=bool, default=False, help='产量归一化')

parser.add_argument('--use_11a2', type=bool, default=True, help='use mod11a2 two branch')
parser.add_argument('--n_head', type=int, default=4, help='transformer config number of head')
parser.add_argument('--n_layers', type=int, default=6, help='number of transformer layers')
parser.add_argument('--cnn_att', type=str, default='cbam', help='[se, cbam]')
parser.add_argument('--kan', type=bool, default=False, help='use kan or mlp')
parser.add_argument('--att', type=bool, default=False, help='attention before cat')
parser.add_argument('--fc', type=bool, default=False, help='kan with fc')
parser.add_argument('--split_str', type=str, default=split_str, help='split_str')
parser.add_argument('--loss_f', type=str, default='huber', help='loss function [mse, l1, rmse, huber]')

parser.add_argument('--struct', type=str, default='two_encoder', help='[one_encoder, two_encoder]')

option = parser.parse_args()
