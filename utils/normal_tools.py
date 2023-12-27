import time
from datetime import datetime
import pytz
import os


def get_format_beijing_time():
    """
    生成基于当前北京时间的字符串，形如：2023-11-03-12-12-59
    """
    tz = pytz.timezone("Asia/Shanghai")
    bj_time = datetime.fromtimestamp(int(time.time()), tz).strftime("%Y-%m-%d-%H-%M-%S")
    # current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return bj_time


def get_folders(project_folder: str, time_now: str):
    """
    返回 background、instance、instance_masks、composite/(时间戳）、composite_mask/(时间戳）文件夹，同时自动创建输出文件夹
    :param project_folder: 整个图片项目的最外层文件夹
    :param time_now: 时间戳（字符串）
    :return:五个字符串，分别为 background、instance、instance_masks、composite/(时间戳）、composite_mask/(时间戳）文件夹
    """
    bg_folder = os.path.join(project_folder, 'backgrounds')
    ins_folder = os.path.join(project_folder, 'instances', 'images')
    ins_mask_folder = os.path.join(project_folder, 'instances', 'masks')
    composite_save_folder = os.path.join(project_folder, 'output', 'composites', time_now)
    comp_mask_save_folder = os.path.join(project_folder, 'output', 'masks', time_now)

    os.makedirs(composite_save_folder, exist_ok=True)
    os.makedirs(comp_mask_save_folder, exist_ok=True)

    return bg_folder, ins_folder, ins_mask_folder, composite_save_folder, comp_mask_save_folder
