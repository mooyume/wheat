import os
import rasterio


def read_tif_metadata(tif_filepath):
    """
    读取 TIFF 文件并获取其 NoData 值、宽度和高度。

    Args:
        tif_filepath (str): TIFF 文件的完整路径。

    Returns:
        tuple: 包含 (nodata值, 宽度, 高度) 的元组。
               如果无法读取文件或获取信息，则返回 None。
    """
    try:
        with rasterio.open(tif_filepath) as dataset:
            nodata = dataset.nodata
            width = dataset.width
            height = dataset.height
            return nodata, width, height
    except rasterio.errors.RasterioIOError as e:
        print(f"无法读取 TIFF 文件: {tif_filepath}, 错误信息: {e}")
        return None
    except Exception as e:
        print(f"读取 TIFF 元数据时发生错误: {e}")
        return None


def get_all_deepest_tif_dimensions_with_max(folder_path):
    """
    递归获取指定文件夹下所有最深层次的目录，并读取每个目录中第一个 .tif 文件，
    获取其宽度和高度并打印，同时记录并打印遇到的最大宽度和高度。

    Args:
        folder_path (str):  要搜索的根文件夹路径。

    Returns:
        None: 函数直接打印结果，没有显式返回值.
              如果找不到最深目录或 .tif 文件，或读取失败，会打印相应的提示信息。
    """
    deepest_dirs = []
    max_depth = -1
    max_width_overall = 0  # 初始化最大宽度
    max_height_overall = 0  # 初始化最大高度

    for root, dirs, files in os.walk(folder_path):
        depth = root.count(os.sep) - folder_path.count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_dirs = [root]  # 发现更深层级，替换之前的目录列表
        elif depth == max_depth and depth > -1:  # 深度相同，添加到列表
            deepest_dirs.append(root)

    if deepest_dirs:
        print(f"最深层次目录 (共 {len(deepest_dirs)} 个):")
        for deepest_dir in deepest_dirs:
            print(f"  目录: {deepest_dir}")
            tif_file_found = False
            for file in os.listdir(deepest_dir):
                if file.lower().endswith(".tif"):
                    tif_filepath = os.path.join(deepest_dir, file)
                    metadata = read_tif_metadata(tif_filepath)
                    if metadata:
                        _, width, height = metadata
                        print(f"    第一个 .tif 文件: {file}")
                        print(f"    宽度: {width}")
                        print(f"    高度: {height}")
                        tif_file_found = True

                        # 更新最大宽度和高度
                        max_width_overall = max(max_width_overall, width)
                        max_height_overall = max(max_height_overall, height)

                        break  # 找到第一个 .tif 文件后停止搜索
                    else:
                        print(f"    读取 .tif 文件 '{file}' 元数据失败。")
                        break  # 读取失败，跳到下一个最深目录
            if not tif_file_found:
                print(f"    在目录 '{deepest_dir}' 中未找到 .tif 文件。")

        if max_width_overall > 0 or max_height_overall > 0:  # 只有当至少找到一个tif文件时才打印最大值
            print(f"\n所有最深目录的 .tif 文件中:")
            print(f"  最大宽度: {max_width_overall}")
            print(f"  最大高度: {max_height_overall}")
        else:
            print("\n在任何最深目录中都没有找到有效的 .tif 文件，无法确定最大宽度和高度。")

    else:
        print(f"在文件夹 '{folder_path}' 下未找到任何子目录。")



def get_deepest_directories(folder_path):
    """
    递归获取指定文件夹下所有最深层次的目录路径。

    Args:
        folder_path (str):  要搜索的根文件夹路径。

    Returns:
        list: 包含最深层次目录路径的列表。
               如果未找到任何子目录，则返回空列表。
    """
    deepest_dirs = []
    max_depth = -1

    for root, dirs, files in os.walk(folder_path):
        depth = root.count(os.sep) - folder_path.count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_dirs = [root] # 发现更深层级，替换之前的目录列表
        elif depth == max_depth and depth > -1: # 深度相同，添加到列表
            deepest_dirs.append(root)

    return deepest_dirs



def process_tifs_in_deepest_dirs_dynamic_output(input_folder):
    """
    获取指定文件夹下最深层次的目录，并处理这些目录中的所有 TIFF 文件：
    1. 将 NoData 值修改为 0。
    2. 使用 LZW 压缩格式压缩。
    3. 将处理后的 TIFF 文件保存到修改后的最深目录路径下，
       路径修改方式为将最深目录路径中的 "guanzhong" 替换为 "guanzhong111"。

    Args:
        input_folder (str):  输入文件夹的路径。

    Returns:
        None: 函数直接进行文件操作，没有显式返回值。
              会打印处理过程中的信息和错误提示。
    """
    deepest_dirs = get_deepest_directories(input_folder)

    if not deepest_dirs:
        print(f"在文件夹 '{input_folder}' 下未找到任何子目录，无法处理 TIFF 文件。")
        return

    print(f"找到 {len(deepest_dirs)} 个最深层次目录，开始处理 TIFF 文件...")

    for deepest_dir in deepest_dirs:
        print(f"\n处理目录: {deepest_dir}")

        # 基于最深目录路径构建输出目录路径，并替换 "guanzhong"
        output_dir_path_modified = deepest_dir.replace("gz", "gz11") # 替换路径中的 "guanzhong"

        if not os.path.exists(output_dir_path_modified):
            os.makedirs(output_dir_path_modified, exist_ok=True) # 确保修改后的输出目录存在

        for file in os.listdir(deepest_dir):
            if file.lower().endswith((".tif", ".tiff")):
                input_filepath = os.path.join(deepest_dir, file)
                output_filepath = os.path.join(output_dir_path_modified, file) # 使用修改后的输出目录路径

                try:
                    with rasterio.open(input_filepath, 'r') as src:
                        profile = src.profile.copy()
                        data = src.read()
                        original_nodata = profile.get('nodata') # 获取原始nodata值，用于打印信息

                        profile.update(dtype=data.dtype, nodata=0, compress='lzw') # 更新 nodata 和 压缩

                        with rasterio.open(output_filepath, 'w', **profile) as dst:
                            dst.write(data)
                        print(f"  已处理文件: {input_filepath}")
                        print(f"    - NoData 值已修改为: 0 (原 NoData: {original_nodata})")
                        print(f"    - 压缩格式已设置为: LZW")
                        print(f"    - 保存到: {output_filepath} (路径已修改)")

                except rasterio.errors.RasterioIOError as e:
                    print(f"  无法读取 TIFF 文件: {input_filepath}, 错误信息: {e}")
                except Exception as e:
                    print(f"  处理文件时发生错误: {input_filepath}, 错误信息: {e}")



def process_tifs_recursive_dtype(folder_path):
    """
    递归处理指定文件夹下所有子文件夹中的 TIFF 文件，
    修改 NoData 值为 0，并将像素类型和像素深度修改为有符号整型 16 位。

    Args:
        folder_path (str):  要处理的根文件夹路径。

    Returns:
        None: 函数直接修改文件，没有显式返回值。
              会打印处理过程中的信息和错误提示。
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                filepath = os.path.join(root, file)
                try:
                    with rasterio.open(filepath, 'r+') as dataset: # 使用 'r+' 模式直接修改文件
                        original_nodata = dataset.nodata # 获取原始 NoData 值，用于打印信息

                        dataset.nodata = 0 # 设置 NoData 值为 0
                        # dataset.profile.update(dtype=rasterio.int16) # 更新像素类型为 int16

                        data = dataset.read() # 读取数据 (需要重新读取以应用 dtype 更改，或者在write时astype)
                        dataset.write(data.astype(rasterio.int16)) # 将数据写回，并转换为 int16

                        print(f"已处理文件: {filepath}")
                        print(f"  - NoData 值已修改为: 0 (原 NoData: {original_nodata})")
                        # print(f"  - 像素类型已修改为: 有符号整型 16 位 (int16)")

                except rasterio.errors.RasterioIOError as e:
                    print(f"无法读取 TIFF 文件: {filepath}, 错误信息: {e}")
                except Exception as e:
                    print(f"处理文件时发生错误: {filepath}, 错误信息: {e}")

# 调用示例
if __name__ == '__main__':
    # tif_file = r"E:\25holiday\Dataset\guanzhong\output\mcd12q1_mask\guanzhong\2003\610102\2002-10-08.tif"  # 替换为你的 TIFF 文件路径
    # metadata = read_tif_metadata(tif_file)
    #
    # if metadata:
    #     nodata_value, width, height = metadata
    #     print(f"文件: {tif_file}")
    #     print(f"  NoData 值: {nodata_value}")
    #     print(f"  宽度: {width}")
    #     print(f"  高度: {height}")
    # else:
    #     print(f"无法获取文件 '{tif_file}' 的元数据。")

    # example_root_folder = r"E:\25holiday\Dataset\guanzhong\output\mcd12q1_mask\guanzhong\2002"  # 替换为你要操作的实际根文件夹路径
    # folder_to_process = example_root_folder  # 替换为你想要处理的文件夹路径
    # get_all_deepest_tif_dimensions_with_max(folder_to_process)

    # input_folder = r"E:\25holiday\Dataset\guanzhong\output\fldas\gz"  # 替换为你的输入文件夹路径
    # process_tifs_in_deepest_dirs_dynamic_output(input_folder)


    input_folder = r"E:\25holiday\Dataset\guanzhong\mod09a1\custom_mask"
    process_tifs_recursive_dtype(input_folder)
