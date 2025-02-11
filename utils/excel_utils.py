import pandas as pd


def calculate_yearly_counts(file_path):
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
    df['年份'] = df['年份'].astype(int)

    # 按年份列聚合，计算每个年份的行数
    year_counts = df['年份'].value_counts().sort_index()

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


if __name__ == '__main__':
    # 示例调用
    file_path = r'E:\25holiday\Dataset\label\weinan.xlsx'
    result = calculate_yearly_counts(file_path)
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
    input_excel = r'E:\25holiday\Dataset\label\weinan-convert.xlsx'
    plot_output_folder_only_plot = 'output_yield_plots/weinan'
    plot_yield_by_region(input_excel, plot_output_folder_only_plot)
