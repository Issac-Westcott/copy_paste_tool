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


def get_scaled_image(ins_img, bg_img, args, bg_img_dict: dict = None, specified_classes: list = None):
    """
    :param ins_img:
    :param bg_img:
    :param args:
    :param bg_img_dict:
    :param specified_classes: 一个列表，指定参考哪些类的大小作为自动缩放参照。
    比如想要粘贴小汽车目标，应参考small-vehicle而非airport的大小
    :return:
    """
    # assert (0 < args.min_scaling_factor <= args.max_scaling_factor <= 1) or args.bg_json_path is not None

    if (not args.manual_scaling) and (bg_img_dict is not None):
        if specified_classes is None:
            specified_classes = bg_img_dict["exist_category"]
        specified_boxes_dict = {category: bg_img_dict[category] for category in specified_classes}
        average_width, average_height = calculate_average_size(specified_boxes_dict)

        # Calculate the scaling factor based on the average size
        min_scaling_factor = min(average_width / ins_img.width, average_height / ins_img.height)
        max_scaling_factor = max(average_width / ins_img.width, average_height / ins_img.height)
    else:
        min_scaling_factor = args.min_scaling_factor
        max_scaling_factor = args.max_scaling_factor

    scaling_factor = random.uniform(min_scaling_factor, max_scaling_factor)

    # Scale the new object image
    new_width = int(ins_img.width * scaling_factor)
    new_height = int(ins_img.height * scaling_factor)
    scaled_ins_image = ins_img.resize((new_width, new_height))

    return scaled_ins_image


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


def find_non_overlapping_position(new_width, new_height, image_size, existing_bounding_boxes):
    max_attempts = 1000  # You can adjust the maximum number of attempts
    if image_size[0] - new_width < 0 or image_size[1] - new_height < 0:
        return None, None
    for _ in range(max_attempts):
        x = random.randint(0, image_size[0] - new_width)
        y = random.randint(0, image_size[1] - new_height)

        new_box = [
            [x, y, x + new_width, y, x + new_width, y + new_height, x, y + new_height]
        ]

        # Check for overlap with existing bounding boxes
        overlap = any(is_overlap(new_box, existing_bounding_boxes[category]) for category in existing_bounding_boxes)

        if not overlap:
            return x, y

    # If no non-overlapping position is found, return None
    return None, None


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
