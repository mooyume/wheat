import pandas as pd

import pandas as pd

def get_filtered_data():
    # 读取原始 Excel 文件中的数据
    input_file = r'E:\25holiday\data\area\crop-area-gansu.xlsx'
    data = pd.read_excel(input_file)

    # 根据条件筛选数据
    filtered_data = data[(data['所属地级市'].isin(['天水市', '平凉市', '陇南市','平凉地区','平凉市','陇南地区'])) & (data['农作物种类或名称'] == '小麦')]

    # 将筛选后的数据写入新的 Excel 文件
    output_file = r'E:\25holiday\data\area\crop-area-filtered.xlsx'
    filtered_data.to_excel(output_file, index=False)

    print('筛选后的数据已保存到', output_file)




def spe():
    import pandas as pd

    # 读取 Excel 文件
    input_file = r'E:\25holiday\data\area\crop-area-gansu.xlsx'
    data = pd.read_excel(input_file, sheet_name='data')
    filtered_data = data[(data['所属省份'] == '河南省')]

    # 选择指定的列，例如 '农作物种类或名称'
    filtered_data = data['农作物种类或名称']

    # 去重并打印不重复的值
    unique_values = filtered_data.drop_duplicates()

    print('列中不重复的值:')
    for value in unique_values:
        print(value)


if __name__ == '__main__':
    # spe()
    # 调用函数
    get_filtered_data()
