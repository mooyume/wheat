import pandas as pd
from functools import lru_cache

import config.config


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


def filter_yield(file_paths=None):
    try:
        if file_paths is None:
            file_paths = config.config.file_paths

        if not file_paths:
            raise ValueError("文件路径列表为空")

        # 合并 Excel 文件
        merged_df = merge_excel_files(tuple(file_paths))  # 使用 tuple 以便于缓存

        if merged_df.empty:
            raise ValueError("合并后的数据框为空")

        # 过滤数据：按照单位进行不同的过滤条件
        filtered_data = merged_df.copy()
        filtered_data = filtered_data[(
                (filtered_data['单位'] == '吨') & (filtered_data['产量'] < 10000) |
                (filtered_data['单位'] == '万吨') & (filtered_data['产量'] < 1)
        )]

        if filtered_data.empty:
            return {}

        # 按照县域代码作为键，统计年度、产量、单位作为值中的元素列表
        result = {}
        for _, row in filtered_data.iterrows():
            county_code = row['县域代码']
            entry = [row['统计年度'], row['产量'], row['单位']]
            if county_code not in result:
                result[county_code] = []
            result[county_code].append(entry)

        return result

    except Exception as e:
        print(f"发生错误: {e}")
        return {}


if __name__ == "__main__":
    result = filter_yield()
    print(len(result))
    # 打印result，每个元素换行
    for county_code, entries in result.items():
        print(f"县域代码: {county_code}")
        for entry in entries:
            print(f"  统计年度: {entry[0]}, 产量: {entry[1]}, 单位: {entry[2]}")
        print()
