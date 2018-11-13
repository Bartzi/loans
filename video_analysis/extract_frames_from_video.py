import argparse

import csv
import numpy as np

import cv2
import os
from PIL import Image
from tqdm import tqdm

IMAGE_TYPES = [".png", ".jpg", ".jpeg"]


def extract_frames(video_path, output_path, resize_max=None):
    video_reader = cv2.VideoCapture(video_path)
    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for frame_num in tqdm(range(number_of_frames)):
            if not video_reader.isOpened():
                break

            ret, frame = video_reader.read()
            if ret is False:
                break

            b, g, r = np.split(frame, 3, axis=2)
            frame = np.concatenate((r, g, b), axis=2)
            image = Image.fromarray(frame)

            if resize_max is not None:
                scale_factor = resize_max / max(image.size)

                new_size = [min(int(round(scale_factor * dim)), resize_max) for dim in image.size]
                image = image.resize(new_size, Image.LANCZOS)

            image.save(os.path.join(output_path, f'{frame_num}.png'))
    finally:
        video_reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the HPI Sheep in a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input_videos", nargs='+', default=[], help="path to recorded video that shall be analyzed")
    parser.add_argument("output", help="path to output directory of extracted frames")
    parser.add_argument("-r", "--resize-max", type=int, help="max size of one side should be this size")
    parser.add_argument("--recreate-gt", action='store_true', default=False, help="do not exract images only recreate gt file")

    args = parser.parse_args()

    if not args.recreate_gt:
        for video_path in tqdm(args.input_videos):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(args.output, video_name)
            os.makedirs(output_path, exist_ok=True)

            extract_frames(video_path, output_path, resize_max=args.resize_max)

    # create groundtruth csv file
    with open(os.path.join(args.output, 'gt.csv'), 'w') as handle:
        writer = csv.writer(handle, delimiter='\t')

        for path, _, file_names in os.walk(args.output):
            for file_name in filter(lambda x: os.path.splitext(x)[-1].lower() in IMAGE_TYPES, file_names):
                writer.writerow([os.path.join(os.path.relpath(path, args.output), file_name)])

