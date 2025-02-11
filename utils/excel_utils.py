import pandas as pd


def calculate_yearly_counts(file_path, col_name='统计年份'):
    """
    读取 Excel 文件，按年份列聚合并计算每个年份的行数。

    参数:
    file_path (str): Excel 文件的路径。

    返回:
    pandas.Series: 每个年份的行数统计结果。
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 确保年份列中的所有数据都是整数
    df[col_name] = df[col_name].astype(str)

    # 按年份列聚合，计算每个年份的行数
    year_counts = df[col_name].value_counts().sort_index()

    return year_counts


import matplotlib.pyplot as plt


def process_excel_file(file_path, output_path):
    """
    读取 Excel 文件，处理播种面积和产量的单位转换，并计算每公顷的产量，
    插入新列 'kg/公顷'，同时修改产量单位和播种面积单位为转换后的值，并将结果写入新的 Excel 文件。

    参数:
    file_path (str): 输入的 Excel 文件路径。
    output_path (str): 输出的 Excel 文件路径。

    返回:
    pandas.DataFrame: 包含新列 'kg/公顷' 的数据框。
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 单位转换
    df['播种面积'] = df.apply(lambda row: row['播种面积'] / 15 if row['播种面积单位'] == '亩' else row['播种面积'], axis=1)
    df['播种面积单位'] = '公顷'

    df['产量'] = df.apply(lambda row: row['产量'] * 1000 if row['产量单位'] == '吨' else row['产量'], axis=1)
    df['产量单位'] = 'kg'

    # 计算产量除以播种面积，并插入新列
    df['kg/公顷'] = df['产量'] / df['播种面积']

    # 将结果写入新的 Excel 文件
    df.to_excel(output_path, index=False)

    return df


def plot_kg_per_hectare(file_path):
    """
    读取 Excel 文件，按地区分组并按年份排序，然后为每个地区单独绘制 kg/公顷 折线图。
    在绘图前打印每个地区的键和值，值为按年份排序后的（年份，产量）元组列表。

    参数:
    file_path (str): 输入的 Excel 文件路径。

    返回:
    None
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 按地区分组后按年份排序
    grouped = df.groupby('地区')

    # 打印每个地区的键和值
    for name, group in grouped:
        sorted_group = group.sort_values(by='年份')
        print(
            f'{name}: {len(list(zip(sorted_group["年份"], sorted_group["kg/公顷"])))}{list(zip(sorted_group["年份"], sorted_group["kg/公顷"]))}')

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_group['年份'], sorted_group['kg/公顷'], marker='o', label=name)

        plt.xlabel('年份')
        plt.ylabel('kg/公顷')
        plt.title(f'{name} 的每公顷产量变化')
        plt.legend()
        plt.grid(True)
        plt.show()


def convert_excel_units_and_calculate_ratio(excel_file_path, output_file_path):
    """
    读取 Excel 文件，转换产量和播种面积单位，计算单位面积产量（千克/平方千米），
    并将结果保存到新的 Excel 文件。

    参数:
        excel_file_path (str): 输入 Excel 文件的路径。
        output_file_path (str): 输出 Excel 文件的路径。
    """

    def convert_yield(row):
        """
        嵌套函数：转换产量单位为千克。
        """
        yield_value = row['产量']
        yield_unit = row['产量单位']

        if pd.isna(yield_unit):
            return yield_value
        elif yield_unit == '吨':
            return yield_value * 1000
        elif yield_unit == '千克':
            return yield_value
        else:
            print(f"未知产量单位: {yield_unit}，行索引: {row.name}，将不进行转换。请检查您的数据。")
            return yield_value

    def convert_area(row):
        """
        嵌套函数：转换播种面积单位为平方千米。
        """
        area_value = row['播种面积']
        area_unit = row['播种面积单位']

        if pd.isna(area_unit):
            return area_value
        elif area_unit == '公顷':
            return area_value * 0.01
        elif area_unit == '亩':
            return area_value * (1 / 1500)
        elif area_unit == '平方千米':
            return area_value
        else:
            print(f"未知播种面积单位: {area_unit}，行索引: {row.name}，将不进行转换。请检查您的数据。")
            return area_value

    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file_path)

        # 应用转换函数并创建新列
        df['产量(千克)'] = df.apply(convert_yield, axis=1)
        df['播种面积(平方千米)'] = df.apply(convert_area, axis=1)

        # 计算单位面积产量 (千克/平方千米) 并创建新列
        df['单位面积产量(千克/平方千米)'] = df.apply(lambda row: row['产量(千克)'] / row['播种面积(平方千米)']
        if row['播种面积(平方千米)'] != 0 and not pd.isna(row['播种面积(平方千米)'])
        else 0, axis=1)  # 避免除以0和处理NaN值

        # 保存修改后的 DataFrame 到新的 Excel 文件
        df.to_excel(output_file_path, index=False)

        print(f"已完成单位转换和单位面积产量计算，结果已保存到文件: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件未找到: {excel_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")


def custom_linear_interpolate(series):
    """
    自定义线性插值函数，用于填充 pandas Series 中的 NaN 值。

    参数:
        series (Series):  需要进行插值的 pandas Series (数值型).

    返回:
        Series: 插值后的 pandas Series.
    """
    interpolated_series = series.copy()  # 创建 Series 的副本，避免修改原始数据
    nan_indices = interpolated_series.index[interpolated_series.isnull()].tolist()  # 获取 NaN 值的索引列表

    if not nan_indices:  # 如果没有 NaN 值，直接返回原始 Series
        return interpolated_series

    valid_indices = interpolated_series.index[~interpolated_series.isnull()].tolist()  # 获取非 NaN 值的索引列表

    for nan_index in nan_indices:
        lower_valid_index = None
        upper_valid_index = None

        # 寻找 Nan 值索引之前的最近有效值索引
        for valid_idx in reversed(valid_indices):  # 逆序遍历，找到前面最近的
            if valid_idx < nan_index:
                lower_valid_index = valid_idx
                break

        # 寻找 Nan 值索引之后的最近有效值索引
        for valid_idx in valid_indices:  # 正序遍历，找到后面最近的
            if valid_idx > nan_index:
                upper_valid_index = valid_idx
                break

        if lower_valid_index is not None and upper_valid_index is not None:
            # 线性插值计算
            x0, y0 = lower_valid_index, interpolated_series[lower_valid_index]
            x1, y1 = upper_valid_index, interpolated_series[upper_valid_index]
            interpolated_value = y0 + (
                    interpolated_series.index.get_loc(nan_index) - interpolated_series.index.get_loc(x0)) * (
                                         y1 - y0) / (
                                         interpolated_series.index.get_loc(x1) - interpolated_series.index.get_loc(
                                     x0))
            interpolated_series[nan_index] = interpolated_value  # 填充插值结果
        elif lower_valid_index is not None:
            # 只有下界有效值，向前填充 (forward fill)
            interpolated_series[nan_index] = interpolated_series[lower_valid_index]
        elif upper_valid_index is not None:
            # 只有上界有效值，向后填充 (backward fill)
            interpolated_series[nan_index] = interpolated_series[upper_valid_index]
        # 如果上下界都没有有效值，则保持 NaN (或者您可以根据需求设置为其他值，例如 0)

    return interpolated_series


def interpolate_missing_yield_data(excel_file_path):
    """
    读取 Excel 文件，按地区分组和年份排序，对单位面积产量为 0 的数据进行插值，
    并将插值结果更新回 DataFrame，并打印更新后的结果。

    参数:
        excel_file_path (str): 输入 Excel 文件的路径。
    """

    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_file_path)

        # 检查 '单位面积产量(千克/平方千米)' 列是否存在，如果不存在则提示用户并返回
        if '单位面积产量(千克/平方千米)' not in df.columns:
            print("错误: Excel 文件中缺少 '单位面积产量(千克/平方千米)' 列。请确保列名正确，并先运行单位转换和计算脚本生成该列。")
            return

        # 按照地区列分组，再按照年份排序
        df_grouped = df.groupby('地区', group_keys=False).apply(lambda x: x.sort_values(by='年份'))

        # 遍历每个地区的数据
        interpolated_regions_data = []  # 用于存储插值后的地区数据
        for region, region_data in df_grouped.groupby('地区'):
            # 强制将 '单位面积产量(千克/平方千米)' 列转换为数值类型
            region_data['单位面积产量(千克/平方千米)'] = pd.to_numeric(region_data['单位面积产量(千克/平方千米)'], errors='coerce')

            # 使用线性插值，替换单位面积产量为 0 的值
            # 先将 0 值替换为 NaN，以便 interpolate() 函数处理
            region_data['单位面积产量(千克/平方千米)'] = region_data['单位面积产量(千克/平方千米)'].replace(0, pd.NA)

            # **使用自定义的线性插值函数进行插值**
            region_data['单位面积产量(千克/平方千米)'] = custom_linear_interpolate(region_data['单位面积产量(千克/平方千米)'])

            # **修改后的打印部分：打印插值后的 region_data 的 '单位面积产量(千克/平方千米)' 列**
            print(f"\n----- 地区: {region}，插值后的 '单位面积产量(千克/平方千米)' 列 (使用自定义插值)-----")
            print(region_data[['年份', '单位面积产量(千克/平方千米)']].to_string())  # 只打印年份和单位面积产量列，并使用 to_string() 完整显示
            print("----- 地区数据打印结束 -----\n")

        # 将插值后的各个地区数据合并为一个 DataFrame
        df_interpolated = pd.concat(interpolated_regions_data)

        # **注意：此时 df_interpolated 已经包含了插值后的 '单位面积产量(千克/平方千米)' 列**

        # 如果需要，可以将 df_interpolated 保存到新的 Excel 文件 (取消注释以下代码)
        # output_excel_file = 'interpolated_excel_file_custom.xlsx' # 修改了输出文件名
        # df_interpolated.to_excel(output_excel_file, index=False)
        # print(f"插值结果已保存到文件: {output_excel_file}")


    except FileNotFoundError:
        print(f"错误: 文件未找到: {excel_file_path}")
    except KeyError as e:
        print(f"错误: Excel 文件缺少列: {e}。请检查列名是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")


import os


def plot_yield_by_region(excel_file_path, output_folder='output_plots',
                         interpolated_df=None):  # 新增绘图方法，新增 interpolated_df 参数
    """
    读取 Excel 文件，按地区分组，绘制每个地区的单位面积产量折线图，
    并将图表保存到指定文件夹。

    参数:
        excel_file_path (str): 输入 Excel 文件的路径。
        output_folder (str):  保存图表的文件夹路径，默认为 'output_plots'。
        interpolated_df (DataFrame, 可选): 已经插值好的 DataFrame，如果提供，则直接使用该 DataFrame 绘图，
                                         否则，函数会读取 excel_file_path 并进行插值。
    """
    try:
        # 如果没有提供插值好的 DataFrame，则读取 Excel 文件并进行分组排序 (绘图函数也可以独立使用)
        if interpolated_df is None:
            df = pd.read_excel(excel_file_path)
            # 检查 '单位面积产量(千克/平方千米)' 列是否存在
            if '单位面积产量(千克/平方千米)' not in df.columns:
                print("错误: Excel 文件中缺少 '单位面积产量(千克/平方千米)' 列。请确保列名正确，并先运行单位转换和计算脚本生成该列。")
                return
            df_grouped = df.groupby('地区', group_keys=False).apply(lambda x: x.sort_values(by='年份'))
        else:
            df_grouped = interpolated_df.groupby('地区', group_keys=False).apply(
                lambda x: x.sort_values(by='年份'))  # 如果提供了插值好的数据，也需要分组

        # 确保输出文件夹存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历每个地区的数据进行绘图
        for region, region_data in df_grouped.groupby('地区'):
            # **绘图代码 (与之前绘图函数中的代码相同)**
            plt.figure(figsize=(10, 6))
            plt.plot(region_data['年份'], region_data['单位面积产量(千克/平方千米)'], marker='o', linestyle='-')
            plt.title(f'{region} - 单位面积产量 (千克/平方千米)')
            plt.xlabel('年份')
            plt.ylabel('单位面积产量 (千克/平方千米)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_file_path = os.path.join(output_folder, f'{region}_yield_plot.png')
            plt.savefig(plot_file_path)
            plt.close()

            print(f"已为地区 {region} 生成折线图，并保存到: {plot_file_path}")

        print(f"已完成所有地区的折线图绘制，图表已保存到文件夹: {output_folder}")


    except FileNotFoundError:
        print(f"错误: 文件未找到: {excel_file_path}")
    except KeyError as e:
        print(f"错误: Excel 文件缺少列: {e}。请检查列名是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")


def fill_district_code(excel_path, sheet_name='宝鸡', district_column='县域名称', district_code_column='研究地区'):
    """
    读取 Excel 文件，当某一行 '地区' 列和 '地区编码' 列都不为空时，
    将 '地区' 列作为 key，'地区编码' 列作为 value 存储到字典中。
    当 '地区编码' 列为空时，按照 '地区' 列为 key 从字典中获取 value，
    写入空的 '地区编码' 位置。

    Args:
        excel_path (str): Excel 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        district_column (str, optional): 包含地区名称的列名，默认为 '地区'。
        district_code_column (str, optional): 包含地区编码的列名，默认为 '地区编码'。

    Returns:
        pandas.DataFrame:  填充了地区编码列的 Pandas DataFrame。
                           如果发生错误，则返回 None 并打印错误信息。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{excel_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return None

    district_code_map = {}  # 初始化字典用于存储地区和地区编码

    for index, row in df.iterrows():
        district_name = row[district_column]
        district_code = row[district_code_column]

        if pd.notna(district_name) and pd.notna(district_code):  # 地区列和地区编码列都不为空
            district_code_map[district_name] = district_code
        elif pd.isna(district_code) and pd.notna(district_name):  # 地区编码列为空，地区列不为空
            if district_name in district_code_map:
                df.loc[index, district_code_column] = district_code_map[district_name]

    return df


def convert_unit_area_yield(excel_path, sheet_name='Sheet1', input_column='单位面积产量(千克/平方千米)',
                            output_column='单位面积产量(千克/公顷)'):
    """
    读取 Excel 文件，根据 '单位面积产量(千克/平方千米)' 列计算新的 '单位面积产量(千克/公顷)' 列，
    单位转换：千克/平方千米  ->  千克/公顷 (除以 100)。

    Args:
        excel_path (str): Excel 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        input_column (str, optional):  输入列名 (单位面积产量 千克/平方千米)，默认为 '单位面积产量(千克/平方千米)'。
        output_column (str, optional): 输出列名 (单位面积产量 千克/公顷)，默认为 '单位面积产量(千克/公顷)'。

    Returns:
        pandas.DataFrame:  添加了 '单位面积产量(千克/公顷)' 列的 Pandas DataFrame。
                           如果发生错误，则返回 None 并打印错误信息。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{excel_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return None

    if input_column not in df.columns:
        print(f"错误: 输入列 '{input_column}' 在 Excel 文件中未找到。")
        return None

    # 单位转换： 1 平方千米 = 100 公顷
    conversion_factor = 100

    df[output_column] = df[input_column] / conversion_factor

    return df


def convert_xls_to_xlsx(xls_path, xlsx_path):
    """
    读取 XLS 文件数据并写入 XLSX 文件。

    Args:
        xls_path (str):  输入 XLS 文件的路径。
        xlsx_path (str): 输出 XLSX 文件的路径。

    Returns:
        bool: True 如果转换成功，False 如果发生错误。
    """
    try:
        # 读取 XLS 文件
        df = pd.read_excel(xls_path)

        # 写入 XLSX 文件
        df.to_excel(xlsx_path, index=False)  # index=False 避免写入 DataFrame 索引

        return True  # 转换成功

    except FileNotFoundError:
        print(f"错误: XLS 文件 '{xls_path}' 未找到。")
        return False
    except Exception as e:
        print(f"转换 XLS 到 XLSX 文件时发生错误: {e}")
        return False


def delete_rows_by_district(excel_path, sheet_name='Sheet1', district_column='地区',
                            districts_to_delete=['新城区', '碑林区', '莲湖区', '农工商', '未央区', '雁塔区', '西咸新区']):
    """
    读取 Excel 文件，当 '地区' 列的值为指定列表中的任何一个时，删除该行。

    Args:
        excel_path (str): Excel 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        district_column (str, optional): 包含地区名称的列名，默认为 '地区'。
        districts_to_delete (list, optional):  要删除的地区名称列表，默认为 ['新城区', '碑林区', '莲湖区']。

    Returns:
        pandas.DataFrame:  删除了指定行后的 Pandas DataFrame。
                           如果发生错误，则返回 None 并打印错误信息。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{excel_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return None

    initial_row_count = len(df)  # 记录初始行数

    # 使用 isin() 方法和布尔索引删除行
    df = df[~df[district_column].isin(districts_to_delete)]  # ~ 表示取反，保留不在 districts_to_delete 中的行

    deleted_row_count = initial_row_count - len(df)  # 计算删除的行数

    print(f"已删除 {deleted_row_count} 行 (地区为: {districts_to_delete})。")

    return df


def calculate_yield_kg_per_hectare(excel_path, sheet_name='Sheet1', sowing_area_column='播种面积-公顷',
                                   total_yield_column='总产量-吨', output_column='单位面积产量(千克/公顷)'):
    """
    读取 Excel 文件，根据 '播种面积-公顷' 和 '总产量-吨' 列，计算新的 '单位面积产量(千克/公顷)' 列。
    单位转换：总产量从吨转换为千克 (乘以 1000)。

    Args:
        excel_path (str): Excel 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        sowing_area_column (str, optional):  播种面积列名 (公顷)，默认为 '播种面积-公顷'。
        total_yield_column (str, optional): 总产量列名 (吨)，默认为 '总产量-吨'。
        output_column (str, optional): 输出列名 (单位面积产量 千克/公顷)，默认为 '单位面积产量(千克/公顷)'。

    Returns:
        pandas.DataFrame:  添加了 '单位面积产量(千克/公顷)' 列的 Pandas DataFrame。
                           如果发生错误，则返回 None 并打印错误信息。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{excel_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return None

    if sowing_area_column not in df.columns:
        print(f"错误: 播种面积列 '{sowing_area_column}' 在 Excel 文件中未找到。")
        return None

    if total_yield_column not in df.columns:
        print(f"错误: 总产量列 '{total_yield_column}' 在 Excel 文件中未找到。")
        return None

    # 单位转换： 1 吨 = 1000 千克
    ton_to_kg_factor = 1000

    # 计算单位面积产量 (千克/公顷)
    df[output_column] = (df[total_yield_column] * ton_to_kg_factor) / df[sowing_area_column]

    return df


def extract_unique_county_codes_to_txt(excel_paths, output_txt_path, sheet_name='Sheet1', county_code_column='县域代码'):
    """
    读取指定的三个 Excel 文件，提取 '县域代码' 列，去重后写入 txt 文档。

    Args:
        excel_paths (list): 包含三个 Excel 文件路径的列表。
        output_txt_path (str): 输出 txt 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        county_code_column (str, optional):  包含县域代码的列名，默认为 '县域代码'。

    Returns:
        bool: True 如果操作成功，False 如果发生错误。
    """
    unique_county_codes = set()  # 使用 set 自动去重

    for excel_path in excel_paths:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            if county_code_column not in df.columns:
                print(f"警告: Excel 文件 '{excel_path}' 中未找到列 '{county_code_column}'，跳过该文件。")
                continue  # 跳过当前文件，处理下一个
            county_codes = df[county_code_column].astype(str).tolist()  # 读取县域代码列并转换为字符串列表
            unique_county_codes.update(county_codes)  # 添加到 set 中，自动去重

        except FileNotFoundError:
            print(f"错误: Excel 文件 '{excel_path}' 未找到，操作终止。")
            return False  # 遇到文件未找到错误，直接终止
        except Exception as e:
            print(f"读取 Excel 文件 '{excel_path}' 时发生错误: {e}，操作终止。")
            return False  # 遇到其他错误，直接终止

    try:
        with open(output_txt_path, 'w') as txtfile:
            for code in sorted(list(unique_county_codes)):  # 排序后写入，可选
                txtfile.write(code + '\n')
        return True  # 操作成功

    except Exception as e:
        print(f"写入 txt 文件 '{output_txt_path}' 时发生错误: {e}。")
        return False  # 写入 txt 文件错误


def normalize_unit_area_yield(excel_path, sheet_name='Sheet1', yield_column='单位面积产量(千克/公顷)',
                              normalized_column='单位面积产量(归一化)'):
    """
    读取 Excel 文件，对 '单位面积产量(千克/公顷)' 列进行 Min-Max 归一化，并将归一化后的值写入新的列。

    Args:
        excel_path (str): Excel 文件的路径。
        sheet_name (str, optional): 要读取的 Excel 工作表名称，默认为 'Sheet1'。
        yield_column (str, optional):  单位面积产量列名 (千克/公顷)，默认为 '单位面积产量(千克/公顷)'。
        normalized_column (str, optional): 归一化后的列名，默认为 '单位面积产量(归一化)'。

    Returns:
        pandas.DataFrame:  添加了归一化列的 Pandas DataFrame。
                           如果发生错误，则返回 None 并打印错误信息。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{excel_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取 Excel 文件时发生错误: {e}")
        return None

    if yield_column not in df.columns:
        print(f"错误: 单位面积产量列 '{yield_column}' 在 Excel 文件中未找到。")
        return None

    # 获取要归一化的列
    yield_data = df[yield_column]

    # 计算最小值和最大值
    min_yield = yield_data.min()
    max_yield = yield_data.max()

    # 避免除以零的情况 (如果最大值和最小值相等，则所有值都相同，归一化后都为 0 或 1，这里设置为 0)
    if max_yield == min_yield:
        df[normalized_column] = 0.0  # 或者设置为 1.0，根据你的需求
        print(f"警告: 单位面积产量列的最大值和最小值相等，归一化结果全部为 0。")
    else:
        # Min-Max 归一化公式
        df[normalized_column] = (yield_data - min_yield) / (max_yield - min_yield)

    return df


if __name__ == '__main__':
    # 示例调用
    file_path = r'E:\25holiday\data\label\guanzhong-label.xlsx'
    result = calculate_yearly_counts(file_path, col_name='县域代码')
    print(result)

    # 示例调用
    # file_path = r'E:\25holiday\Dataset\label\weinan.xlsx'
    # out_path = r'E:\25holiday\Dataset\label\weinan-.xlsx'
    # result_df = process_excel_file(file_path, out_path)
    # print(result_df)
    # plot_kg_per_hectare(r'E:\25holiday\Dataset\label\baoji-1.xlsx')

    # 单位转换
    # input_excel = r'E:\25holiday\Dataset\label\weinan.xlsx'
    # output_excel = r'E:\25holiday\Dataset\label\weinan-convert.xlsx'
    # convert_excel_units_and_calculate_ratio(input_excel, output_excel)

    # 插值
    # interpolate_missing_yield_data(r'E:\25holiday\Dataset\label\baoji-interpolated.xlsx')

    #  仅绘图，不插值，图表保存到默认文件夹 'output_plots'
    # input_excel = r'E:\25holiday\Dataset\label\weinan-convert.xlsx'
    # plot_output_folder_only_plot = 'output_yield_plots/weinan'
    # plot_yield_by_region(input_excel, plot_output_folder_only_plot)

    # excel_file_path = r'E:\25holiday\data\label\baoji-label.xlsx'  # 替换为你的 Excel 文件路径
    # updated_df = fill_district_code(excel_file_path)
    #
    # if updated_df is not None:
    #     output_excel_path = r'E:\25holiday\data\label\baoji-label-1.xlsx' # 可以保存到新文件，或者覆盖原文件
    #     try:
    #         updated_df.to_excel(output_excel_path, index=False) # 保存到新的 Excel 文件，不包含索引列
    #         print(f"已将填充地区编码后的 Excel 文件保存到: '{output_excel_path}'")
    #     except Exception as e:
    #         print(f"保存 Excel 文件时发生错误: {e}")

    # 单位转换
    # excel_file_path = r'E:\25holiday\data\label\weinan-label.xlsx'  # 替换为你的 Excel 文件路径
    # updated_df = convert_unit_area_yield(excel_file_path)
    #
    # if updated_df is not None:
    #     output_excel_path = r'E:\25holiday\data\label\weinan-label-1.xlsx'  # 可以保存到新文件，或者覆盖原文件
    #     try:
    #         updated_df.to_excel(output_excel_path, index=False)  # 保存到新的 Excel 文件，不包含索引列
    #         print(f"已将添加单位面积产量(千克/公顷)列的 Excel 文件保存到: '{output_excel_path}'")
    #     except Exception as e:
    #         print(f"保存 Excel 文件时发生错误: {e}")

    # input_xls_file = r'E:\25holiday\data\咸阳_end.xls'  # 替换为你的 XLS 文件路径
    # output_xlsx_file = r'E:\25holiday\data\xianyang.xlsx' # 替换为你想要保存的 XLSX 文件路径
    #
    # conversion_successful = convert_xls_to_xlsx(input_xls_file, output_xlsx_file)

    # excel_file_path = r'E:\25holiday\data\xian.xlsx'
    # updated_df = delete_rows_by_district(excel_file_path)
    #
    # if updated_df is not None:
    #     output_excel_path = excel_file_path # 可以保存到新文件，或者覆盖原文件
    #     try:
    #         updated_df.to_excel(output_excel_path, index=False) # 保存到新的 Excel 文件，不包含索引列
    #         print(f"已将删除指定行后的 Excel 文件保存到: '{output_excel_path}'")
    #     except Exception as e:
    #         print(f"保存 Excel 文件时发生错误: {e}")

    # excel_file_path = r'E:\25holiday\data\xianyang.xlsx'  # 替换为你的 Excel 文件路径
    # updated_df = calculate_yield_kg_per_hectare(excel_file_path)
    #
    # if updated_df is not None:
    #     output_excel_path = r'E:\25holiday\data\label\xianyang-label.xlsx' # 可以保存到新文件，或者覆盖原文件
    #     try:
    #         updated_df.to_excel(output_excel_path, index=False) # 保存到新的 Excel 文件，不包含索引列
    #         print(f"已将添加单位面积产量(千克/公顷)列的 Excel 文件保存到: '{output_excel_path}'")
    #     except Exception as e:
    #         print(f"保存 Excel 文件时发生错误: {e}")

    # excel_file_paths = [
    #     r'E:\25holiday\data\label\xianyang-label.xlsx',  # 替换为你的第一个 Excel 文件路径
    #     r'E:\25holiday\data\label\xian-label.xlsx',  # 替换为你的第二个 Excel 文件路径
    #     r'E:\25holiday\data\label\baoji-label.xlsx',   # 替换为你的第三个 Excel 文件路径
    #     r'E:\25holiday\data\label\weinan-label.xlsx'   # 替换为你的第三个 Excel 文件路径
    # ]
    # output_txt_file = '../data/code/guanzhong.txt'  # 替换为你想要保存的 txt 文件路径
    #
    # success = extract_unique_county_codes_to_txt(excel_file_paths, output_txt_file)
    #
    # if success:
    #     print(f"已将去重后的县域代码写入到 txt 文件 '{output_txt_file}'")
    # else:
    #     print(f"提取唯一县域代码并写入 txt 文件失败。请检查错误信息。")



    excel_file_path = '../data/his_data/guanzhong-label.xlsx'  # 替换为你的 Excel 文件路径
    updated_df = normalize_unit_area_yield(excel_file_path)

    if updated_df is not None:
        output_excel_path = '../data/his_data/guanzhong-label.xlsx' # 可以保存到新文件，或者覆盖原文件
        try:
            updated_df.to_excel(output_excel_path, index=False) # 保存到新的 Excel 文件，不包含索引列
            print(f"已将添加归一化单位面积产量列的 Excel 文件保存到: '{output_excel_path}'")
        except Exception as e:
            print(f"保存 Excel 文件时发生错误: {e}")
