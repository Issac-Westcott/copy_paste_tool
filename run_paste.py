import argparse
import os
import random

from tqdm import tqdm
import json
from utils.normal_tools import *
from utils.image_tools import *
from PIL import Image
import warnings


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_folder", type=str, default="/home/npz/workspace/copy_paste_tool/example",
                        help="存放所有素材的文件夹路径。具体文件夹格式见主程序")
    parser.add_argument("--ins_category", type=list, default=None,
                        help="需要粘贴的目标图像的类别。要求将不同类目标按类存放在instances文件夹下。详见主程序")
    parser.add_argument("--gen_num", type=int, default=10,
                        help="如果大于0，则模式变为强制生成多少张合成图片，背景会反复随机选取")
    parser.add_argument("--min_num_ins_per_bg", type=int, default=1, help="设定每张图片上放置的最小目标个数")
    parser.add_argument("--max_num_ins_per_bg", type=int, default=7, help="设定每张图片上放置的最大目标个数")
    parser.add_argument("--bg_json_path", type=str, default=None,
                        help="如果是在已有目标的图上粘贴新目标，为了防止随机选取粘贴坐标导致的遮挡，可以读取原图的标注数据提取已有bbox"
                             "以防止遮挡，同时可以根据原有物体平均大小自动设定缩放scale，防止物体过大/过小"
                             "json格式见example/example_bg_data.json。可以为None")
    parser.add_argument("--manual_scaling", type=bool, default=False,
                        help="如果为False，默认尝试读取原数据json自动获取scaling，只有找不到json才使用设定缩放比例；"
                             "如果为True，则强制使用人工设定的缩放上下限")
    parser.add_argument("--min_scaling_factor", type=float, default=0.1,
                        help="设定每张图片上放置目标的缩放比例下限（0-1）。"
                             "若为1，则代表instance的长/宽此时与背景长/宽相等（以先到达100%者为准")
    parser.add_argument("--max_scaling_factor", type=float, default=0.15,
                        help="设定每张图片上放置目标的缩放比例下限（0-1）"
                             "若为1，则代表instance的长/宽此时与背景长/宽相等（以先到达100%者为准")
    parser.add_argument("--no_ins_mask", type=bool, default=False,
                        help="如果为True，则确定没有instance masks，paste时不会使用mask，而是直接整个贴上去")
    parser.add_argument("--ins_dominant", type=bool, default=False,
                        help="若instance数据较少，可以将此参数设为True。这会让程序改为遍历所有instance，一旦全部粘贴完即退出，"
                             "一定程度上防止过拟合（默认情况为遍历所有背景）。")
    parser.add_argument("--ins_num_ceiling", type=int, default=-1,
                        help="随机选取最多多少张目标图片进行copy&paste，适合小范围测试。值为-1时则无影响")
    parser.add_argument("--bg_num_ceiling", type=int, default=-1,
                        help="随机选取最多多少张背景图片进行copy&paste，适合小范围测试。值为-1时则无影响")
    parser.add_argument("--output_format", default='json', choices=['json', 'DOTA'],
                        help="输出数据格式")
    parser.add_argument("--classes_for_autoscaling", type=list, default=None,
                        help="一个列表，指定参考哪些类的大小作为自动缩放参照。"
                             "比如想要粘贴小汽车目标，应参考small-vehicle而非airport的大小")
    parser.add_argument("--max_attempt_finding_xy", type=int, default=1000000,
                        help="寻找不重叠的粘贴坐标时，最大允许的尝试次数。避免死循环。若找不到则跳过这个instance")
    parser.add_argument("--resample_method", default="LANCZOS", choices=['LANCZOS', 'BILINEAR', 'BICUBIC'],
                        help="图像缩放时的插值方法")
    parser.add_argument("--yolo_class_list", type=list,
                        default=['car', 'truck', 'tank', 'armored_car', 'radar', 'artillery', 'boat', 'airplane'])
    parser.add_argument("--motion_mode", type=bool, default=False,
                        help="是否生成有规律运动的目标，只允许有一个物体")
    parser.add_argument("--controlnet_gen_data", type=bool, default=False,
                        help="controlnet生成数据时，文件夹格式有所不同，mask映射关系也会改变")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    project_folder下文件格式：
        ./args.project_folder_path
                    |---backgrounds
                    |---instances
                            |---(args.ins_category[0])
                                |---images
                                |---masks(optional)
                            |---...
                    |---output(相关文件夹会自动创建)
                            |---时间戳
                                |---composites
                                    |---...（图片）
                                |---masks
                                    |---...（图片）
                                |---labels

    json格式参考example_bg_data.json。需要注意，如果列表中某个dict内没有“img_class”一项，检索该图片时则会在backgrounds文件夹内遍历，
    如果有重名情况可能会找到错误的图片

    默认情况下，目标图像多，背景图像少。在每张背景图上随机放置给定数量上下限内的目标图像（两张合成图内，目标可能重复）。
    其他情况下的合成正在实现
    """
    args = get_parser()
    time_now = get_format_beijing_time()

    # 目标图片文件名要和相应的mask文件名有对应关系，如instance图片文件名“P008.png”，
    # 对应mask图片命名需要同样是“P008.png”，或“P008_mask.png”（mask后缀名随意，可以为png、jpg或jpeg，无需对应）
    # 如果有需要也可以自己再指定目录
    (bg_folder, ins_folder_list, ins_mask_folder_list,
     composite_save_folder, comp_mask_save_folder, composite_label_folder) = get_folders(
        args.project_folder,
        time_now,
        args.ins_category,
        args.no_ins_mask
    )

    # 得到全部目标图片路径和背景文件路径，同时分别计数
    # 如果是controlnet合成数据，instances/{装备类型}/images 文件夹下，每张图片都有一个单独的文件夹.
    # 同时还会有canny.jpg这样的不需要的数据，需要予以剔除
    if args.controlnet_gen_data:
        ins_path_list = []
        for ins_folder in ins_folder_list:
            for ctrl_image_folder in os.listdir(ins_folder):
                for file in os.listdir(os.path.join(ins_folder, ctrl_image_folder)):
                    if "canny" not in file:  # 排除canny.jpg等文件
                        ins_path_list.append(os.path.join(ins_folder, ctrl_image_folder, file))
        ins_num = len(ins_path_list)
    else:
        ins_path_list = []
        # ins_folder_list: 所有类的images文件夹绝对路径
        for ins_folder in ins_folder_list:
            for file in os.listdir(ins_folder):
                ins_path_list.append(os.path.join(ins_folder, file))
        ins_num = len(ins_path_list)

    bg_path_list = [os.path.join(bg_folder, file) for file in os.listdir(bg_folder)]
    bg_num = len(bg_path_list)

    # 如果是在已有标注的数据上paste，读取原始背景中的数据
    # if args.bg_json_path is not None:
    original_data = []
    if args.bg_json_path:
        if os.path.exists(args.bg_json_path):
            with open(args.bg_json_path, 'r') as f:
                original_data = json.load(f)  # 背景图片对应json里的原始数据
            assert original_data
    # 开始合成
    used_bg = used_ins = 0

    if args.ins_dominant:
        raise NotImplementedError
    else:
        if args.gen_num <= 0:
            # 该模式下，有多少个背景图片就生成多少张合成图。
            print(f"Walking through all {len(bg_path_list)} images.")
            for bg_path in tqdm(bg_path_list):
                bg_img = Image.open(bg_path)
                ins_num_per_bg = random.randint(args.min_num_ins_per_bg, args.max_num_ins_per_bg)
                selected_ins_path_list = get_some_instances(ins_path_list, ins_num_per_bg)
                bg_data_dict = new_ins_data_dict = {}
                final_mask_img = Image.new('RGB', (bg_img.size[0], bg_img.size[1]), (0, 0, 0))

                # 如果存在bg json数据，则读取
                if original_data:
                    for one_dict in original_data:  # one_dict：每张图对应的字典，内含img_name、instances、exist_category三个key
                        if one_dict['img_name'] == os.path.basename(bg_path):
                            bg_data_dict = one_dict
                            bg_data_dict['bg_img_info'] = {
                                'img_name': os.path.basename(bg_path),
                                'bg_width': bg_img.width,
                                'bg_height': bg_img.height
                            }
                            break
                else:
                    bg_data_dict = {
                        "bg_img_info": {
                            'img_name': os.path.basename(bg_path),
                            'bg_width': bg_img.width,
                            'bg_height': bg_img.height
                        },
                        'instances': {},
                        'exist_category': []
                    }
                have_at_least_one_instance = False  # 防止极端情况下一个合适的坐标都没找到，而生成无目标图
                for ins_path in selected_ins_path_list:
                    ins_mask_path = get_ins_mask_dir(ins_path)
                    ins_img = Image.open(ins_path)
                    ins_mask_img = Image.open(ins_mask_path).convert("L") if (
                            ins_mask_path and not args.no_ins_mask) else None
                    scaled_ins_img, scaled_ins_mask_img = get_scaled_image(
                        ins_img, ins_mask_img, bg_img, args, bg_data_dict)
                    if args.no_ins_mask:
                        scaled_ins_mask_img = None
                    existing_bounding_boxes = bg_data_dict['instances'] if bg_data_dict['instances'] else {}
                    x, y = find_non_overlapping_position(scaled_ins_img.size, bg_img.size,
                                                         existing_bounding_boxes, args.max_attempt_finding_xy)

                    # 准备粘贴
                    if x is None or y is None:
                        continue
                    else:
                        bg_img, bbox = paste_img_or_mask(scaled_ins_img, bg_img, (x, y), scaled_ins_mask_img)
                        final_mask_img, _ = paste_img_or_mask(scaled_ins_mask_img, final_mask_img, (x, y))
                        ins_class_name = os.path.basename(os.path.dirname(os.path.dirname(ins_path)))
                        if ins_class_name in bg_data_dict['instances'].keys():
                            bg_data_dict['instances'][ins_class_name].append(bbox)
                        else:
                            bg_data_dict['instances'][ins_class_name] = [bbox]
                            bg_data_dict['exist_category'].append(ins_class_name)

                        have_at_least_one_instance = True
                if have_at_least_one_instance:
                    # 将时间戳转化为月份日期格式。如"1227"
                    datetime_obj = datetime.strptime(time_now, "%Y-%m-%d-%H-%M-%S")
                    month_date_str = datetime_obj.strftime("%m%d")

                    saved_img_num = len(os.listdir(composite_save_folder))
                    bg_img.save(
                        os.path.join(composite_save_folder, f"composite_{month_date_str}_{str(saved_img_num + 1)}.png"))
                    if args.no_ins_mask:
                        final_mask_img.save(os.path.join(
                            comp_mask_save_folder,
                            f"composite_{month_date_str}_{str(saved_img_num + 1)}_mask.png")
                        )
                    os.makedirs(os.path.join(composite_label_folder, 'json'), exist_ok=True)
                    os.makedirs(os.path.join(composite_label_folder, 'yolo_txt'), exist_ok=True)
                    with open(os.path.join(
                            composite_label_folder, 'json',
                            f"composite_{month_date_str}_{str(saved_img_num + 1)}.json"), 'w') as f:
                        json.dump(bg_data_dict, f, indent=4)

                    json_to_yolov8(bg_data_dict, os.path.join(composite_label_folder, 'yolo_txt',
                                                              f"composite_{month_date_str}_{str(saved_img_num + 1)}.txt"),
                                   args.yolo_class_list)
                else:
                    warnings.warn(f"背景图 {os.path.basename(bg_path)} 中难以粘贴合适的instance，请留意（该图像未保存）",
                                  UserWarning)
        else:
            print(f"Generating {args.gen_num} images, as set in args.gen_num")
            if args.motion_mode:  # 如果需要生成有轨迹运动的图片的话（目标数量只能为1）
                motion_path_coor = generate_motion_path(args.gen_num)
            for ii in tqdm(range(args.gen_num)):
                bg_path = random.choice(bg_path_list)
                bg_img = Image.open(bg_path)
                ins_num_per_bg = random.randint(args.min_num_ins_per_bg, args.max_num_ins_per_bg)
                selected_ins_path_list = get_some_instances(ins_path_list, ins_num_per_bg)
                bg_data_dict = new_ins_data_dict = {}
                final_mask_img = Image.new('RGB', (bg_img.size[0], bg_img.size[1]), (0, 0, 0))

                # 如果存在bg json数据，则读取
                if original_data:
                    for one_dict in original_data:  # one_dict：每张图对应的字典，内含img_name、instances、exist_category三个key
                        if one_dict['img_name'] == os.path.basename(bg_path):
                            bg_data_dict = one_dict
                            bg_data_dict['bg_img_info'] = {
                                'img_name': os.path.basename(bg_path),
                                'bg_width': bg_img.width,
                                'bg_height': bg_img.height
                            }
                            break
                else:
                    bg_data_dict = {
                        "bg_img_info": {
                            'img_name': os.path.basename(bg_path),
                            'bg_width': bg_img.width,
                            'bg_height': bg_img.height
                        },
                        'instances': {},
                        'exist_category': []
                    }
                have_at_least_one_instance = False  # 防止极端情况下一个合适的坐标都没找到，而生成无目标图

                for ins_path in selected_ins_path_list:
                    if args.controlnet_gen_data:
                        ins_mask_path = get_ctrlnet_ins_mask_dir(ins_path)
                    else:
                        ins_mask_path = get_ins_mask_dir(ins_path)
                    ins_img = Image.open(ins_path)
                    ins_mask_img = Image.open(ins_mask_path).convert("L") if (
                            ins_mask_path and not args.no_ins_mask) else None
                    # 中间已经包含着mask与image dimension一致的断言
                    scaled_ins_img, scaled_ins_mask_img = get_scaled_image(
                        ins_img, ins_mask_img, bg_img, args, bg_data_dict)
                    existing_bounding_boxes = bg_data_dict['instances'] if bg_data_dict['instances'] else {}
                    x, y = find_non_overlapping_position(scaled_ins_img.size, bg_img.size,
                                                         existing_bounding_boxes, args.max_attempt_finding_xy)
                    if args.motion_mode:
                        x = int(motion_path_coor[ii][0])
                        y = int(motion_path_coor[ii][1])
                    # 准备粘贴
                    if x is None or y is None:
                        continue
                    else:
                        bg_img, bbox = paste_img_or_mask(scaled_ins_img, bg_img, (x, y), scaled_ins_mask_img)
                        if not args.no_ins_mask:
                            final_mask_img, _ = paste_img_or_mask(scaled_ins_mask_img, final_mask_img, (x, y))
                        ins_class_name = os.path.basename(os.path.dirname(os.path.dirname(ins_path)))
                        if ins_class_name in bg_data_dict['instances'].keys():
                            bg_data_dict['instances'][ins_class_name].append(bbox)
                        else:
                            bg_data_dict['instances'][ins_class_name] = [bbox]
                            bg_data_dict['exist_category'].append(ins_class_name)

                        have_at_least_one_instance = True
                if have_at_least_one_instance:
                    # 将时间戳转化为月份日期格式。如"1227"
                    datetime_obj = datetime.strptime(time_now, "%Y-%m-%d-%H-%M-%S")
                    month_date_str = datetime_obj.strftime("%m%d")

                    saved_img_num = len(os.listdir(composite_save_folder))
                    bg_img.save(
                        os.path.join(composite_save_folder, f"composite_{month_date_str}_{str(saved_img_num + 1)}.png"))
                    if not args.no_ins_mask:
                        os.makedirs(comp_mask_save_folder, exist_ok=True)
                        final_mask_img.save(os.path.join(
                            comp_mask_save_folder,
                            f"composite_{month_date_str}_{str(saved_img_num + 1)}_mask.png")
                        )
                    os.makedirs(os.path.join(composite_label_folder, 'json'), exist_ok=True)
                    os.makedirs(os.path.join(composite_label_folder, 'yolo_txt'), exist_ok=True)
                    with open(os.path.join(
                            composite_label_folder, 'json',
                            f"composite_{month_date_str}_{str(saved_img_num + 1)}.json"), 'w') as f:
                        json.dump(bg_data_dict, f, indent=4)

                    json_to_yolov8(bg_data_dict, os.path.join(composite_label_folder, 'yolo_txt',
                                                              f"composite_{month_date_str}_{str(saved_img_num + 1)}.txt"),
                                   args.yolo_class_list)
                else:
                    warnings.warn(f"背景图 {os.path.basename(bg_path)} 中难以粘贴合适的instance，请留意（该图像未保存）",
                                  UserWarning)
    print("Finished")
