from PIL import Image
import os
import random


def calculate_average_size(bounding_boxes):
    total_width = 0
    total_height = 0
    total_boxes = 0

    for category, boxes in bounding_boxes.items():
        for bbox in boxes:
            bbox_tuples = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]
            width = max(coord[0] for coord in bbox_tuples) - min(coord[0] for coord in bbox_tuples)
            height = max(coord[1] for coord in bbox_tuples) - min(coord[1] for coord in bbox_tuples)
            total_width += width
            total_height += height
            total_boxes += 1

    average_width = total_width / total_boxes
    average_height = total_height / total_boxes

    return average_width, average_height


def get_scaled_image(ins_img, mask_img, bg_img, args, bg_data_dict: dict = None):
    """
    :param ins_img:
    :param bg_img:
    :param args:
    :param bg_data_dict:
    :return:
    """
    if (not args.manual_scaling) and bg_data_dict:
        if not args.classes_for_autoscaling:
            specified_classes = bg_data_dict["exist_category"]
        else:
            specified_classes = args.classes_for_autoscaling
        specified_boxes_dict = {category: bg_data_dict["instances"][category]
                                for category in specified_classes}
        average_width, average_height = calculate_average_size(specified_boxes_dict)

        # Calculate the scaling factor based on the average size
        min_scaling_factor = min(average_width / ins_img.width, average_height / ins_img.height)
        max_scaling_factor = max(average_width / ins_img.width, average_height / ins_img.height)
    else:
        min_scaling_factor = args.min_scaling_factor
        max_scaling_factor = args.max_scaling_factor

    assert 0 < args.min_scaling_factor <= args.max_scaling_factor <= 1
    scaling_factor = random.uniform(min_scaling_factor, max_scaling_factor)

    # Scale the new object images
    new_width = int(ins_img.width * scaling_factor)
    new_height = int(ins_img.height * scaling_factor)
    if args.resample_method == 'LANCZOS':
        scaled_ins_image = ins_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        if mask_img:
            scaled_mask_img = mask_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            assert ins_img.size == mask_img.size
    else:
        if args.resample_method == 'BILINEAR':
            scaled_ins_image = ins_img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
            if mask_img:
                scaled_mask_img = mask_img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
                assert ins_img.size == mask_img.size
        else:
            if args.resample_method == 'BICUBIC':
                scaled_ins_image = ins_img.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
                if mask_img:
                    scaled_mask_img = mask_img.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
                    assert ins_img.size == mask_img.size
            else:
                scaled_ins_image = ins_img.resize((new_width, new_height))
                if mask_img:
                    scaled_mask_img = mask_img.resize((new_width, new_height))
                    assert ins_img.size == mask_img.size
    if not mask_img:
        scaled_mask_img = None


    return scaled_ins_image, scaled_mask_img


def visualize_bbox():
    raise NotImplementedError


def is_overlap(box1, box2):
    for coord1 in box1:
        for coord2 in box2:
            x1_min, y1_min, x1_max, y1_max = min(coord1[0::2]), min(coord1[1::2]), max(coord1[0::2]), max(coord1[1::2])
            x2_min, y2_min, x2_max, y2_max = min(coord2[0::2]), min(coord2[1::2]), max(coord2[0::2]), max(coord2[1::2])
            if not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max):
                return True
    return False


def find_non_overlapping_position(ins_img_size, bg_img_size, existing_bounding_boxes, max_attempts):
    max_attempts = 1000 if max_attempts is None else max_attempts  # 调整每个目标粘贴时在背景图片上尝试最大次数

    # 如果目标图比背景图大，全部返回None
    if bg_img_size[0] - ins_img_size[0] < 0 or bg_img_size[1] - ins_img_size[1] < 0:
        return None, None
    for _ in range(max_attempts):
        x = random.randint(0, bg_img_size[0] - ins_img_size[0])
        y = random.randint(0, bg_img_size[1] - ins_img_size[1])

        new_box = [
            [x, y, x + ins_img_size[0], y, x + ins_img_size[0], y + ins_img_size[1], x, y + ins_img_size[1]]
        ]

        # 检查是否与任何现存bbox重叠
        overlap = any(is_overlap(new_box, existing_bounding_boxes[category]) for category in existing_bounding_boxes)

        if not overlap:
            return x, y

    # 如果找不到或者超出最大尝试次数，全部返回None
    return None, None


def get_some_instances(instance_img_path_list, img_num) -> list:
    ins_list = []
    if img_num > len(instance_img_path_list):
        img_num = len(instance_img_path_list)
    random_ins_names = random.sample(instance_img_path_list, img_num)
    for ins_name in random_ins_names:
        ins_list.append(ins_name)

    return ins_list


def paste_img_or_mask(ins_img, bg_img, coord, mask_for_ins=None):
    """
    注意：如果需要粘贴mask，传入一张黑色图片或者已经粘贴了数个mask的黑色图即可
    :param ins_img: 应为缩放后的目标图
    :param bg_img: 需要粘贴的背景图。
    :param coord: (x,y)元组
    :param mask_for_ins: 如果没有，就做最简单的粘贴；如果不为None，就会根据mask粘贴目标
    :return: 合成图像（PIL.Image)
    """
    if mask_for_ins:
        bg_img.paste(ins_img, coord, mask=mask_for_ins)
    else:
        bg_img.paste(ins_img, coord)

    return bg_img

