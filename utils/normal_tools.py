import json
import os
import time
from datetime import datetime

import pytz


def get_format_beijing_time():
    """
    生成基于当前北京时间的字符串，形如：2023-11-03-12-12-59
    """
    tz = pytz.timezone("Asia/Shanghai")
    bj_time = datetime.fromtimestamp(int(time.time()), tz).strftime("%Y-%m-%d-%H-%M-%S")
    # current_time = datetime.now()
    # formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return bj_time


def get_folders(project_folder: str, time_now: str, ins_category: list, no_mask: bool = False):
    """
    返回 background、instance(基于args.ins_category的list)、instance_masks(基于args.ins_category的list)、
    composite、composite_mask、composite_label文件夹，同时自动创建输出文件夹
    :param no_mask: 是否包含mask路径
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
    os.makedirs(composite_label_folder, exist_ok=True)
    if not no_mask:
        os.makedirs(comp_mask_folder, exist_ok=True)

    return bg_folder, ins_folder_list, ins_mask_folder_list, composite_save_folder, comp_mask_folder, composite_label_folder


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


def get_ctrlnet_ins_mask_dir(ins_path):
    ins_name = os.path.basename(os.path.dirname(ins_path))
    mask_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ins_path))), "masks")
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

    raise FileNotFoundError(f"{ins_name}图片的mask未找到")


def json_to_yolov8(json_data, out_path, yolo_class_index_list):
    """
    yolo_class_index_list: 一个列表，包含着所有需要检测的目标，按顺序排列。这样可以确定写入txt时class的编号
    """
    if isinstance(json_data, str):
        with open(json_data, 'r') as f:
            data = json.load(f)
    else:
        assert isinstance(json_data, dict) or isinstance(json_data, list)
        data = json_data

    bg_width = data["bg_img_info"]["bg_width"]
    bg_height = data["bg_img_info"]["bg_height"]

    with open(out_path, 'w') as out_file:
        for category in data["exist_category"]:
            if category in data["instances"]:
                for instance in data["instances"][category]:
                    x_min = min(instance[0], instance[2], instance[4], instance[6])
                    y_min = min(instance[1], instance[3], instance[5], instance[7])
                    x_max = max(instance[0], instance[2], instance[4], instance[6])
                    y_max = max(instance[1], instance[3], instance[5], instance[7])

                    x_center = (x_min + x_max) / 2.0 / bg_width
                    y_center = (y_min + y_max) / 2.0 / bg_height
                    width = (x_max - x_min) / bg_width
                    height = (y_max - y_min) / bg_height

                    if category not in yolo_class_index_list:
                        raise AttributeError(f"需要在args.yolo_class_list中按照yolo目标index顺序添加{category}类")
                    out_file.write(f"{yolo_class_index_list.index(category)} {x_center} {y_center} {width} {height}\n")
