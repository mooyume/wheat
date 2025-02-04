import pandas as pd
from functools import lru_cache
import time

import config.config


@lru_cache(maxsize=1)
def merge_excel_files(file_paths):
    """
    合并多个 Excel 文件。

    参数:
    - file_paths: 包含多个 Excel 文件路径的元组

    返回:
    - 合并后的 DataFrame
    """
    dfs = []
    for file in file_paths:
        try:
            df = pd.read_excel(file)
            dfs.append(df)
        except FileNotFoundError:
            print(f"警告: 文件 {file} 不存在，跳过该文件")
        except Exception as e:
            print(f"警告: 读取文件 {file} 时发生错误: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_yield_history(input_year, county_code, file_paths=None):
    """
    根据输入的统计年度和县域代码，返回历年产量数据。

    参数:
    - input_year: 统计年度（如 2019）
    - county_code: 县域代码（如 "410181"）
    - file_paths: 包含多个 Excel 文件路径的列表，默认为 None，表示使用默认路径

    返回:
    - 历年产量数据列表，按年份排序，只包含产量
    """
    if file_paths is None:
        file_paths = config.config.file_paths

    # 合并 Excel 文件
    merged_df = merge_excel_files(tuple(file_paths))  # 使用 tuple 以便于缓存

    try:
        input_year = int(input_year)
        county_code = int(county_code)
    except ValueError:
        print("输入的统计年度或县域代码无效，请检查输入")
        return []

    # 过滤数据：年份小于输入年份
    filtered_data = merged_df[merged_df['统计年度'] < input_year].copy()

    # 进一步过滤县域代码匹配的数据
    filtered_data = filtered_data[filtered_data['县域代码'] == county_code]

    # 如果单位是“吨”，转换为“万吨”
    filtered_data.loc[filtered_data['单位'] == '吨', '产量'] = filtered_data.loc[filtered_data['单位'] == '吨', '产量'] / 10000

    # 提取需要的列并排序
    filtered_data = filtered_data[['统计年度', '产量']].sort_values(by='统计年度')

    # 仅返回产量列
    result = filtered_data['产量'].tolist()

    return result


# 示例调用
if __name__ == "__main__":
    input_year = '2020'
    county_code = "410181"

    # 记录开始时间
    start_time = time.time()

    # 循环调用 get_yield_history 方法 100 次
    for _ in range(100):
        result = get_yield_history(input_year, county_code)
        # print(result)  # 如果需要打印每次的结果，可以取消注释

    # 记录结束时间
    end_time = time.time()

    # 计算并打印总运行时间
    total_time = end_time - start_time
    print(f"100 次调用 get_yield_history 方法的总运行时间: {total_time:.4f} 秒")
