import pandas as pd


def merge_excel_files(file_paths):
    """
    合并多个 Excel 文件。

    参数:
    - file_paths: 包含多个 Excel 文件路径的列表

    返回:
    - 合并后的 DataFrame
    """
    # 读取所有 Excel 文件并存储到列表中
    dfs = [pd.read_excel(file) for file in file_paths]

    # 合并所有 DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    return merged_df


def get_yield_history(input_year, county_code, file_paths=[r"E:\25holiday\data\label\gansu-label.xlsx",
                                                           r"E:\25holiday\data\label\henan-label.xlsx",
                                                           r"E:\25holiday\data\label\shanxi-label.xlsx"]):
    """
    根据输入的统计年度和县域代码，返回历年产量数据。

    参数:
    - input_year: 统计年度（如 2019）
    - county_code: 县域代码（如 "410181"）
    - file_paths: 包含多个 Excel 文件路径的列表，默认为三个示例文件路径

    返回:
    - 历年产量数据列表，按年份排序，只包含产量
    """
    # 合并 Excel 文件
    merged_df = merge_excel_files(file_paths)

    # 过滤数据：县域代码匹配且年份小于输入年份
    filtered_data = merged_df[(merged_df['县域代码'] == int(county_code)) & (merged_df['统计年度'] < int(input_year))].copy()

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

    result = get_yield_history(input_year, county_code)
    print(result)
