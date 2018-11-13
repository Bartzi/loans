import argparse
import csv
import json
import numpy as np

import os
import random
from PIL import Image
from chainercv.utils import bbox_iou
from tqdm import tqdm


iou_ranges = list(range(20, 105, 5))
iou_index = -1


def iou_crop(image, bbox, crop_width, crop_height, desired_iou):
    bbox = bbox.astype(np.int32)
    if desired_iou < 0.0:
        crop_x = random.randint(0, image.width - crop_width)
        crop_y = random.randint(0, image.height - crop_height)
    else:
        crop_width_start_end_deviation = int(crop_width // 2 * (1.0 - desired_iou))
        crop_height_start_end_deviation = int(crop_height // 2 * (1.0 - desired_iou))
        crop_x = random.randint(
            max(bbox[0] - crop_width_start_end_deviation, 0),
            min(bbox[0] + crop_width_start_end_deviation, image.width - crop_width)
        )
        crop_y = random.randint(
            max(bbox[1] - crop_height_start_end_deviation, 0),
            min(bbox[1] + crop_height_start_end_deviation, image.height - crop_height)
        )

    corners = np.array(
        [
            crop_x,
            crop_y,
            min(crop_x + crop_width, image.width),
            min(crop_y + crop_height, image.width),
        ]
    )
    return corners


def get_iou_crop(image, paste_x, paste_y, stamp):
    global iou_index
    iou_index = (iou_index + 1) % len(iou_ranges)
    desired_iou = min(iou_ranges[iou_index % len(iou_ranges)] / 100, 1.0)

    num_retries = 0
    good_bbox_found = False
    while not good_bbox_found and num_retries < 200:
        paste_bbox = np.array([paste_x, paste_y, paste_x + stamp.width, paste_y + stamp.height])
        paste_bbox_size = paste_bbox[2:] - paste_bbox[:2]
        max_size_deviation = 1.0 - desired_iou

        for _ in range(200):
            if desired_iou < 0.3:
                crop_width = int(min(stamp.width + (1 - desired_iou) * 10 * stamp.width, image.width))
                crop_height = int(min(stamp.height + (1 - desired_iou) * 10 * stamp.height, image.height))
            else:
                crop_width = random.randint(
                    max(int(paste_bbox_size[0] - paste_bbox_size[0] * max_size_deviation), 1),
                    int(paste_bbox_size[0] + paste_bbox_size[0] * max_size_deviation)
                )
                crop_height = random.randint(
                    max(int(paste_bbox_size[1] - paste_bbox_size[1] * max_size_deviation), 1),
                    int(paste_bbox_size[1] + paste_bbox_size[1] * max_size_deviation)
                )

            crop_bbox = iou_crop(image, paste_bbox, crop_width, crop_height, desired_iou)

            ious = bbox_iou(crop_bbox[None, ...], paste_bbox[None, ...])[0]
            largest_iou = abs(np.max(ious))
            if desired_iou - 0.05 < largest_iou <= desired_iou:
                good_bbox_found = True
                break
        num_retries += 1
    if good_bbox_found is False:
        raise ValueError("No Good BBOX Found")
    return image.crop(crop_bbox), ious[0]


def get_naive_zoom(image, paste_x, paste_y, stamp):
    zoom_ratio = random.random() * 10 + 0.3
    crop_width = min(stamp.width + zoom_ratio * stamp.width, image.width)
    crop_height = min(stamp.height + zoom_ratio * stamp.height, image.height)

    width_insert_ratio = random.random()
    height_insert_ratio = random.random()

    insert_max = [min(paste_x, image.width - crop_width), min(paste_y, image.height - crop_height)]
    insert_min = [max(paste_x + stamp.width - crop_width, 0), max(paste_y + stamp.height - crop_height, 0)]

    for i in range(2):
        if insert_max[i] < insert_min[i]:
            insert_max[i] = insert_min[i]

    insert_point = [int(mi + ratio * (ma - mi)) for mi, ma, ratio in zip(insert_min, insert_max, [width_insert_ratio, height_insert_ratio])]

    crop_bbox = [insert_point[0], insert_point[1], insert_point[0] + crop_width, insert_point[1] + crop_height]
    paste_bbox = np.array([paste_x, paste_y, paste_x + stamp.width, paste_y + stamp.height])
    stamp_with_background = image.crop(crop_bbox)

    iou = bbox_iou(np.array(crop_bbox)[None, ...], paste_bbox[None, ...])[0, 0]
    return stamp_with_background, iou


def create_sample(image, stamp, crop_extra=(0, 0, 0, 0), bbox_sizes=None, zoom_mode=False, image_size=None):
    if bbox_sizes is not None:
        bbox_size = random.choice(bbox_sizes)
    else:
        stamp = stamp.resize(
            (
                random.randint(image_size[0] // 15, image_size[0] // 2),  # width
                random.randint(image_size[1] // 15, image_size[1] // 2),  # height
            ),
            Image.LANCZOS
        )

    if image_size:
        scale_factors = [new_size / old_size for new_size, old_size in zip(image_size, image.size)]
        image = image.resize(image_size, Image.LANCZOS)
        if bbox_sizes is not None:
            bbox_size = [int(dim * factor) for dim, factor in zip(bbox_size, scale_factors)]

    if bbox_sizes is not None:
        stamp = stamp.resize(bbox_size, Image.LANCZOS)

    paste_x = random.randint(crop_extra[0], image.width - stamp.width - crop_extra[2])
    paste_y = random.randint(crop_extra[1], image.height - stamp.height - crop_extra[3])

    paste_image = Image.new('RGBA', image.size)
    paste_image.paste(stamp, (paste_x, paste_y))

    image = Image.alpha_composite(image, paste_image)

    if zoom_mode:
        if image_size is None:
            raise ValueError("if you are using zoom mode, image size can not be None")
        if random.random() >= 0.3:
            stamp_with_background = get_iou_crop(image, paste_x, paste_y, stamp)
        else:
            stamp_with_background = get_naive_zoom(image, paste_x, paste_y, stamp)
    else:
        stamp_with_background = image.crop(
            (
                paste_x - crop_extra[0],
                paste_y - crop_extra[1],
                paste_x + stamp.width + crop_extra[2],
                paste_y + stamp.height + crop_extra[3],
            )
        )
    return stamp_with_background


def get_base_bbox_sizes(base_bbox_path):
    with open(base_bbox_path) as handle:
        bbox_data = json.load(handle)

    bboxes = [item['bounding_boxes'] for item in bbox_data]

    bbox_sizes = set()
    for bbox in bboxes:
        for box in bbox:
            bbox_size = (
                box[3] - box[1],  # width
                box[2] - box[0],  # height
            )
            # filter bboxes that are not correct
            if any(x <= 0 for x in bbox_size):
                continue
            bbox_sizes.add(bbox_size)

    return list(bbox_sizes)


def main(args):
    all_images = os.listdir(args.background_image_dir)
    stamps = [Image.open(stamp) for stamp in args.stamps]

    destination_dir = os.path.join(args.destination, 'images')
    os.makedirs(destination_dir, exist_ok=True)

    # parse base bboxes and create template sizes for bboxes
    if args.base_bboxes is not None:
        bbox_sizes = get_base_bbox_sizes(args.base_bboxes)
    else:
        bbox_sizes = None

    images = []
    for i in tqdm(range(args.num_samples)):
        image_path = random.choice(all_images)
        stamp = random.choice(stamps)

        # randomly flip stamps horizontally
        if random.random() >= 0.5:
            stamp = stamp.transpose(Image.FLIP_LEFT_RIGHT)

        try:
            sample = create_sample(
                Image.open(os.path.join(args.background_image_dir, image_path)).convert("RGBA"),
                stamp,
                crop_extra=args.enlarge_region,
                bbox_sizes=bbox_sizes,
                zoom_mode=args.zoom_mode,
                image_size=args.image_size,
            )
        except ValueError:
            continue

        if args.zoom_mode:
            sample, label = sample
        else:
            label = None

        # resize sample to desired size
        sample = sample.resize(args.output_size, Image.LINEAR)
        file_name = f"images/{i}.png"
        sample.save(os.path.join(args.destination, file_name))
        if label is not None:
            images.append([file_name, format(label, ".4f")])
        else:
            images.append([file_name])

    with open(os.path.join(args.destination, "images.csv"), 'w') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerows(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Put the sheep on any place in the input image save the resulting images as template")
    parser.add_argument("background_image_dir", help="directory that contains all possible background images")
    parser.add_argument("destination", help="destination path for saving images and accompanying gt file (the script creates a new subdir in this directory)")
    parser.add_argument("--stamps", required=True, nargs='+', help="path to search for all sheep stamps that are to be used")
    parser.add_argument("--num-samples", type=int, default=10000, help="number of samples that shall be generated")
    parser.add_argument("--output-size", type=int, nargs=2, default=(75, 75), help="size of output image")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), help="size of image for network (important for zoom mode)")
    parser.add_argument("--enlarge-region", type=int, nargs=4, default=(0, 0, 0, 0), help="number of pixels template image shall be enlarged by")
    parser.add_argument("--base-bboxes", help="base bboxes are used to determine reasonabele sizes of bboxes for sheep pasting, please supply a apth to a json file with all bboxes you want to use")
    parser.add_argument("--zoom-mode", action='store_true', default=False, help="create samples in zoom mode, meaning that stamp is placed on image and then a random zoom is cropped from the image")

    args = parser.parse_args()
    main(args)
