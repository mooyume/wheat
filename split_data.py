import os
import random
from config.config import option as opt
import re
import csv

"""

获取数据集路径，并分割为训练集和验证集

02 - 19  train and validate

20 - 21   test

"""

train_label = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
               '2015', '2016', '2017', '2018']
test_label = ['2019', '2020', '2021']
val_label = ['2019']
all_label = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019', '2020', '2021']


def query_csv(filename, code=None, year=None, region=None):
    match_count = 0
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (code is None or row['code'] == code) and \
                    (year is None or row['year'] == year) and \
                    (region is None or row['region'] == region):
                match_count += 1
    return match_count


def ignore(path):
    region, year, code = get_last_three_parts_separately(path)
    match_count = query_csv('./data/ignore.csv', code, year, region)
    print(f'匹配{match_count}行')
    if match_count > 0:
        return True
    return False


def get_last_three_parts_separately(path):
    parts = path.split("\\")
    return parts[-3], parts[-2], parts[-1]


def split_data(dataset_path, file_path, label, file_name, code_txt_path):
    """
    根据 code 列表筛选目录路径并写入文件，只打印不存在的 code（去重后）。

    Args:
        dataset_path: 数据集根路径。
        file_path: 输出文件路径。
        label: 需要处理的标签列表（例如：['Gansu', 'Shanxi']）。
        file_name: 输出文件名。
        code_txt_path: 包含 code 列表的 txt 文件路径。
    """

    if not os.path.exists(code_txt_path):
        print(f"code 文件不存在：{code_txt_path}")
        return

    try:
        with open(code_txt_path, 'r', encoding='utf-8') as code_file:
            codes = set(line.strip() for line in code_file)
    except FileNotFoundError:
        print(f"找不到 code 文件：{code_txt_path}")
        return
    except Exception as e:
        print(f"读取 code 文件出错：{e}")
        return

    output_file_path = os.path.join(file_path, file_name)
    not_found_codes = set()  # 用于存储不存在的 code，使用 set 自动去重

    if not os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                for r in label:
                    root = os.path.join(dataset_path, r)
                    for subdir, dirs, files in os.walk(root):
                        if subdir == root:
                            continue

                        last_folder = os.path.basename(subdir)
                        if last_folder != 'label':
                            match = re.search(r"\\(\d{6})$", subdir)
                            if match:
                                code = match.group(1)
                                if code in codes:
                                    # 判断ignore文件
                                    if not ignore(subdir):
                                        f.write(subdir + '\n')
                                else:
                                    not_found_codes.add(code)  # 将不存在的 code 添加到 set 中
                            # else: # 不再打印路径格式不匹配的信息
            if not_found_codes:  # 文件写入结束后，判断not_found_codes是否为空，不为空则打印
                print("以下 code 在 code 文件中不存在：")
                for code in not_found_codes:
                    print(code)
        except Exception as e:
            print(f"写入文件出错：{e}")
    else:
        print(f'数据集文件 {file_name} 已存在！')


def split_train_val(input_file, output_file):
    if not os.path.exists(output_file):
        # 读取输入文件的所有行
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # 随机选择大约 1/5 的行
        num_lines = len(lines) // 5
        chosen_lines = random.sample(lines, num_lines)

        # 将选中的行写入输出文件
        with open(output_file, 'w') as f:
            f.writelines(chosen_lines)

        # 从输入文件的行中移除选中的行
        remaining_lines = [line for line in lines if line not in chosen_lines]

        # 将剩余的行写回输入文件
        with open(input_file, 'w') as f:
            f.writelines(remaining_lines)
    else:
        print(f'数据集文件 {output_file} 已存在！')


def build_data(dataset_path, file_path, code_path):
    split_data(dataset_path, file_path, train_label, 'train.txt', code_path)
    split_data(dataset_path, file_path, test_label, 'test.txt', code_path)
    split_data(dataset_path, file_path, val_label, 'val.txt', code_path)
    # split_train_val('./data/train.txt', './data/val.txt')


def build_data_by_code(dataset_path, file_path):
    """
    根据code 排除，  随机24个县不参与训练    生成train.txt 和 val.txt
    :param dataset_path:
    :param file_path:
    :return:
    """
    split_data(dataset_path, file_path, all_label, 'all.txt')
    # 读取txt文件内容
    file_path = './data/all.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 按照 \ 分割每一行，并获取最后一个部分，放入集合中
    unique_last_parts = set(line.strip().split(opt.split_str)[-1] for line in lines)

    # 将集合转换为列表以便随机选择
    unique_last_parts_list = list(unique_last_parts)

    # 随机选出10个放到一个list里
    random_selection = random.sample(unique_last_parts_list, min(24, len(unique_last_parts_list)))
    print(random_selection)

    output_file_a_path = './data/val.txt'
    output_file_b_path = './data/train.txt'

    with open(file_path, 'r') as input_file, \
            open(output_file_a_path, 'w') as output_file_a, \
            open(output_file_b_path, 'w') as output_file_b:

        for line in input_file:
            if any(keyword in line for keyword in random_selection):
                if any(year in line for year in test_label):
                    output_file_a.write(line)
            else:
                if not any(year in line for year in test_label):
                    output_file_b.write(line)


if __name__ == '__main__':
    # build_data('E:\\Crop Prediction\\Datasets\\MOD09A1', './data')
    # build_data(r'E:\25holiday\Dataset\Finally\lzw_mod09a1\Henan', './data', r'E:\25holiday\data\code\henan_code.txt')
    # build_data(r'E:\25holiday\Dataset\Finally\lzw_mod09a1\Henan', './data', r'E:\25holiday\data\code\henan_code.txt')
    # build_data(r'E:\25holiday\Dataset\Finally\lzw_mod09a1\Gansu', './data', r'E:\25holiday\data\code\gansu_code.txt')
    build_data(r'E:\25holiday\Dataset\Finally\lzw_mod09a1\Shanxi', './data', r'./data/code/shanxi_code.txt')
    # build_data(r'/root/Finally/lzw_mod09a1/Gansu', './data', r'./data/code/gansu_code.txt')
    # build_data(r'/root/Finally/lzw_mod09a1/Shanxi', './data', r'./data/code/shanxi_code.txt')
    # build_data(r'/root/Finally/lzw_mod09a1/Henan', './data', r'./data/code/henan_code.txt')
    # build_data_by_code('E:\\project\\dl\\Crop Prediction\\Datasets\\output\\MOD09A1', './data')
