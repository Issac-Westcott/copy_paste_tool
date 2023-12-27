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


def get_folders(project_folder: str, time_now: str, ins_category: list):
    """
    返回 background、instance(基于args.ins_category的list)、instance_masks(基于args.ins_category的list)、
    composite、composite_mask、composite_label文件夹，同时自动创建输出文件夹
    :param project_folder: 整个图片项目的最外层文件夹
    :param time_now: 时间戳（字符串）
    :param ins_category: 准备后续粘贴的类别名称。如果为None则默认将文件夹内全选
    :return:分别为 background(str)、instance(list)、instance_masks(list)、composite_img(str)、
            composite_mask(str) 、composite_label(str)文件夹路径
    """
    if not ins_category:
        ins_category = os.listdir(os.path.join(project_folder, 'instances'))
    bg_folder = os.path.join(project_folder, 'backgrounds')
    ins_folder_list = [os.path.join(project_folder, 'instances', class_name, 'images')
                       for class_name in ins_category]
    ins_mask_folder_list = [os.path.join(project_folder, 'instances', class_name, 'masks')
                            for class_name in ins_category]
    composite_save_folder = os.path.join(project_folder, 'output', time_now, 'composites')
    comp_mask_folder = os.path.join(project_folder, 'output', time_now, 'masks')
    composite_label_folder = os.path.join(project_folder, 'output', time_now, 'labels')

    os.makedirs(composite_save_folder, exist_ok=True)
    os.makedirs(comp_mask_folder, exist_ok=True)
    os.makedirs(composite_label_folder, exist_ok=True)

    return bg_folder, ins_folder_list, ins_mask_folder_list, composite_save_folder, comp_mask_folder


def get_ins_mask_dir(ins_path):
    ins_name = os.path.basename(ins_path)
    mask_folder = os.path.join(os.path.dirname(os.path.dirname(ins_path)), 'masks')
    ins_pref, ins_surf = os.path.splitext(os.path.basename(os.path.join(mask_folder, ins_name)))

    if '_mask' in ins_pref:
        alt_pref = ins_pref.replace('_mask', '')
    else:
        if 'mask' in ins_pref:
            alt_pref = ins_pref.replace('mask', '')
        else:
            alt_pref = ins_pref

    possible_file_surfix = ['.jpg', '.jpeg', '.png']
    possible_mask_surfix = ['', 'mask', '_mask']
    for mask_surfix in possible_mask_surfix:
        for surfix in possible_file_surfix:
            possible_file_path = os.path.join(mask_folder, alt_pref + mask_surfix + surfix)
            if os.path.exists(possible_file_path):
                return possible_file_path

    return None



