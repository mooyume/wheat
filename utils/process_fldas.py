import os
import shutil
import os

"""
    对齐fldas数据
"""
def get_deepest_subfolders(root_dir):
    """
    获取指定文件夹下的所有最深层子文件夹
    :param root_dir: 根目录路径
    :return: 包含所有最深层子文件夹路径的列表
    """
    deepest_folders = []

    def _walk(dir_path):
        # 获取当前目录下的所有子文件夹
        subfolders = [os.path.join(dir_path, d) for d in os.listdir(dir_path)
                      if os.path.isdir(os.path.join(dir_path, d))]

        if not subfolders:  # 如果没有子文件夹，当前目录是最深层
            deepest_folders.append(dir_path)
        else:  # 如果有子文件夹，递归遍历
            for folder in subfolders:
                _walk(folder)

    # 开始遍历
    _walk(root_dir)
    return deepest_folders




def get_year_month(filename):
    """从文件名提取年份和月份，返回(year, month)元组"""
    try:
        name_part = os.path.splitext(filename)[0]  # 去除扩展名
        date_parts = name_part.split('-')
        if len(date_parts) < 3:
            return None
        year = date_parts[0]
        month = date_parts[1].zfill(2)  # 确保两位数月份
        return (year, month)
    except:
        return None


def process_files(a_dir, b_dir, output_dir):
    # 创建输出目录（A目录下的子文件夹）
    os.makedirs(output_dir, exist_ok=True)

    # 构建A目录的年月字典
    a_files = {}
    for fname in os.listdir(a_dir):
        if not fname.lower().endswith('.tif'):
            continue
        ym = get_year_month(fname)
        if ym:
            a_files[ym] = os.path.join(a_dir, fname)

    # 处理B目录文件
    for b_fname in os.listdir(b_dir):
        if not b_fname.lower().endswith('.tif'):
            continue

        ym = get_year_month(b_fname)
        if not ym:
            continue

        if ym in a_files:
            src = a_files[ym]
            dest = os.path.join(output_dir, b_fname)  # 目标路径指向A的子文件夹
            try:
                shutil.copy2(src, dest)
                print(f"Copied: {src} -> {dest}")
            except Exception as e:
                print(f"Error copying {src} to {dest}: {str(e)}")
        else:
            print(f"No match in A for {b_fname} ({ym[0]}-{ym[1]})")


if __name__ == "__main__":

    root_folder = input("请输入根目录路径：").strip()
    deepest_folders = get_deepest_subfolders(root_folder)
    print("最深层子文件夹：")
    for folder in deepest_folders:
        print(folder)
        output_folder = os.path.join(folder, "matched_temporal")  # 自动创建子文件夹
        b_folder = folder.replace("lzw_fldas", "lzw_mod11a2")
        process_files(folder, b_folder, output_folder)
