import os

import pandas as pd


def get_name_by_path(path):
    code = path.split('/')[-1]
    # 读取CSV文件
    df = pd.read_csv('./data/dt_code_name.csv')
    df['dt_adcode'] = df['dt_adcode'].astype(str).str.strip()

    # 按照某一列进行筛选
    filtered_df = df[df['dt_adcode'] == code]
    return filtered_df


def remove_item(pred, real, sowing_area, paths=None, value=3000):
    filtered_pred = []
    filtered_real = []
    filtered_area = []
    filtered_path = []
    removed_paths = []

    for i, (p, r, q, s) in enumerate(zip(pred, real, sowing_area, paths)):
        if abs(p - r) <= value:
            filtered_pred.append(p)
            filtered_real.append(r)
            filtered_area.append(q)
            filtered_path.append(s)
        else:
            if paths is not None and len(paths) > 0:
                removed_paths.append(paths[i])
                filtered_df = get_name_by_path(paths[i])
                # print(f'移除：{filtered_df["dt_name"].values[0]}')

    # 计算移除的元素数量
    k = len(pred) - len(filtered_pred)

    # print(f"移除了 {k} 个元素")
    if paths is not None and len(paths) > 0:
        # print(f"移除的元素对应的paths: {removed_paths}")
        pass

    print(len(filtered_real))

    return filtered_pred, filtered_real, filtered_area, filtered_path, k


def append_string_to_file(string, file_path):
    try:
        # 获取文件的目录路径
        directory = os.path.dirname(file_path)

        # 如果目录不存在，则创建目录
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 追加字符串到文件并换行
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(string + '\n')
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


def process_files(line, df):
    # 将line转换为字符串
    line = str(line)

    # 筛选匹配的行
    matched_rows = df[df['县域代码'].astype(str) == line]

    # 按‘统计年度’列进行排序
    sorted_rows = matched_rows.sort_values(by='统计年度')

    # 只保留需要的列，并重命名列
    output_df = sorted_rows[['统计年度', '产量']]

    return output_df['产量'].tolist()


def main(txt_file, excel_file):
    # 读取excel文件
    df = pd.read_excel(excel_file)

    # 结果字典
    results = {}

    # 逐行读取txt文件
    with open(txt_file, 'r') as file:
        for line in file:
            line = line.strip()
            production_list = process_files(line, df)
            results[line] = production_list

    return results


def save_to_txt(map_data, output_txt):
    with open(output_txt, 'w') as file:
        for key, value in map_data.items():
            value_str = ','.join(map(str, value))
            file.write(f'{key} {value_str}\n')


def load_from_txt(input_txt):
    map_data = {}
    with open(input_txt, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            key = parts[0]
            value = list(map(float, parts[1].split(',')))
            map_data[key] = value
    return map_data


if __name__ == '__main__':
    # 使用示例
    txt_file = r'E:\25holiday\data\code\gansu_code.txt'
    excel_file = r'E:\25holiday\data\label\gansu-label.xlsx'
    output_txt = 'gansu.txt'

    result_map = main(txt_file, excel_file)
    save_to_txt(result_map, output_txt)

    map_data = load_from_txt(output_txt)
    # 打印结果
    for key, value in map_data.items():
        print(f'Code: {key}, Production: {value}')
