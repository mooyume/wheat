import logging
import os
import datetime


def setup_logger(log_path='./log'):
    # 获取当前的年月日时分秒
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d %H-%M-%S')  # 格式化时间字符串
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(os.path.join(log_path, now_str), mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
