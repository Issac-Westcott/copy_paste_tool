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
    parser.add_argument("--project_folder", type=str, default="./example",
                        help="存放所有素材的文件夹路径。具体文件夹格式见主程序")
    parser.add_argument("--ins_category", type=list, default=['boat', 'airplane'],
                        help="需要粘贴的目标图像的类别。要求将不同类目标按类存放在instances文件夹下。详见主程序")
    parser.add_argument("--min_num_ins_per_bg", type=int, default=3, help="设定每张图片上放置的最小目标个数")
    parser.add_argument("--max_num_ins_per_bg", type=int, default=5, help="设定每张图片上放置的最大目标个数")
    parser.add_argument("--bg_json_path", type=str, default="example/example_bg_data.json",
                        help="如果是在已有目标的图上粘贴新目标，为了防止随机选取粘贴坐标导致的遮挡，可以读取原图的标注数据提取已有bbox"
                             "以防止遮挡，同时可以根据原有物体平均大小自动设定缩放scale，防止物体过大/过小"
                             "json格式见example/example_bg_data.json。可以为None")
    parser.add_argument("--manual_scaling", type=bool, default=False,
                        help="如果为False，默认尝试读取原数据json自动获取scaling，只有找不到json才使用设定缩放比例；"
                             "如果为True，则强制使用人工设定的缩放上下限")
    parser.add_argument("--min_scaling_factor", type=float, default=0.2,
                        help="设定每张图片上放置目标的缩放比例下限（0-1）。"
                             "若为1，则代表instance的长/宽此时与背景长/宽相等（以先到达100%者为准")
    parser.add_argument("--max_scaling_factor", type=float, default=0.3,
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
    # 对应mask图片命名需要同样是“P008.png”，或“P008_mask.png”
    (bg_folder, ins_folder_list, ins_mask_folder_list,
     composite_save_folder, comp_mask_save_folder, composite_label_folder) = get_folders(
        args.project_folder,
        time_now,
        args.ins_category
    )

    # 得到全部目标图片路径和背景文件路径，同时分别计数
    ins_path_list = []
    ins_num = 0
    for ins_folder in ins_folder_list:
        for file in os.listdir(ins_folder):
            ins_path_list.append(os.path.join(ins_folder, file))
            ins_num += 1
    bg_path_list = [os.path.join(bg_folder, file) for file in os.listdir(bg_folder)]
    bg_num = len(os.listdir(bg_folder))

    # 如果是在已有标注的数据上paste，读取原始背景中的数据
    # if args.bg_json_path is not None:
    original_data = []
    if os.path.exists(args.bg_json_path):
        with open(args.bg_json_path, 'r') as f:
            original_data = json.load(f)  # 背景图片对应json里的原始数据
        assert original_data
    # 开始合成
    used_bg = used_ins = 0

    if args.ins_dominant:
        raise NotImplementedError
    else:
        for bg_path in bg_path_list:
            bg_img = Image.open(bg_path)
            ins_num_per_bg = random.randint(args.min_num_ins_per_bg, args.max_num_ins_per_bg)
            selected_ins_path_list = get_some_instances(ins_path_list, ins_num_per_bg)
            bg_data_dict  = new_ins_data_dict = {}
            final_mask_img = Image.new('RGB', (bg_img.size[0], bg_img.size[1]), (0, 0, 0))
            if original_data:
                for one_dict in original_data:  # one_dict：每张图对应的字典，内含img_name、instances、exist_category三个key
                    if one_dict['img_name'] == os.path.basename(bg_path):
                        bg_data_dict = one_dict
            have_at_least_one_instance = False  # 防止极端情况下一个合适的坐标都没找到，而生成无目标图
            for ins_path in selected_ins_path_list:
                ins_mask_path = get_ins_mask_dir(ins_path)
                ins_img = Image.open(ins_path)
                ins_mask_img = Image.open(ins_mask_path) if ins_mask_path else None
                scaled_ins_img, scaled_ins_mask_img = get_scaled_image(
                    ins_img, ins_mask_img, bg_img, args, bg_data_dict)
                x, y = find_non_overlapping_position(scaled_ins_img.size, bg_img.size,
                                                     bg_data_dict['instances'], args.max_attempt_finding_xy)

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
                bg_img.save(os.path.join(composite_save_folder, f"{str(len(os.listdir(composite_save_folder)) + 1)}.png"))
                final_mask_img.save(os.path.join(
                    comp_mask_save_folder,
                    f"{str(len(os.listdir(composite_save_folder)) + 1)}_mask.png")
                )
                with open(os.path.join(
                        composite_label_folder,
                        f"{str(len(os.listdir(composite_save_folder)) + 1)}.json"), 'w') as f:
                    json.dump(bg_data_dict, f, indent=4)
            else:
                warnings.warn(f"背景图 {os.path.basename(bg_path)} 中难以粘贴合适的instance，请留意（该图像未保存）", UserWarning)

    print("Finished")
