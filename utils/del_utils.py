import os


def recursive_delete_non_tif(folder_path):
    """
    递归删除指定文件夹及其子文件夹中所有非 .tif 格式的文件。

    Args:
        folder_path (str): 文件夹的路径。

    Returns:
        None: 此函数不返回任何值。删除的文件操作直接在文件系统上进行。
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith(".tif"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")  # 可选：打印已删除的文件路径
                except Exception as e:
                    print(f"删除文件失败: {file_path}, 错误信息: {e}")  # 可选：打印删除失败的文件和错误信息


if __name__ == '__main__':
    recursive_delete_non_tif(r'E:\25holiday\Dataset\guanzhong\output\mcd12q1_mask\guanzhong')
