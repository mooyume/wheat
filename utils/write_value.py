import pandas as pd

# 读取Excel文件A和B
file_A1 = r'E:\25holiday\data\label\henan-label.xlsx'
file_A2 = r'E:\25holiday\data\label\gansu-label.xlsx'
file_A3= r'E:\25holiday\data\label\shanxi-label.xlsx'

df_A1 = pd.read_excel(file_A1)
df_A2 = pd.read_excel(file_A2)
df_A3 = pd.read_excel(file_A3)

# 合并三个数据框
df_A = pd.concat([df_A1, df_A2, df_A3], ignore_index=True)

# 读取Excel文件B
file_B = r'feature.xlsx'
df_B = pd.read_excel(file_B)

# 确保B中有空白列用于存储A的‘单位’和‘产量’列
df_B['单位'] = ''
df_B['产量'] = ''

# 遍历A的每一行，获取年份和地区
for _, row_A in df_A.iterrows():
    year_A = row_A['统计年度']
    region_A = row_A['县域代码']
    unit_A = row_A['单位']
    yield_A = row_A['产量']

    # 在B中进行匹配
    matching_rows_B = df_B[(df_B['Year'] == year_A) & (df_B['Region'] == region_A)]

    # 如果匹配到了行，则在B的空白列写入A的‘单位’和‘产量’列
    for index_B, _ in matching_rows_B.iterrows():
        df_B.at[index_B, '单位'] = unit_A
        df_B.at[index_B, '产量'] = yield_A

# 保存更新后的B文件
output_file_B = 'feature-1.xlsx'
df_B.to_excel(output_file_B, index=False, engine='openpyxl')

print(f"数据已更新并保存至 {output_file_B}")

