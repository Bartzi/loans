import argparse
import numpy as np
import os
import json
from tqdm import tqdm

from PIL import Image

from sheeping.sheep_localizer import SheepLocalizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the HPI Sheep in images",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("log_file", help="path to log file that has been used to train model")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-i", "--images", metavar="IMAGE", nargs="+", help="images to search through")
    source_group.add_argument("-j", "--json", help="json file which contains paths to images")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="id of gpu to use")
    parser.add_argument("-t", "--score-threshold", type=float, default=0.3, help="when to recognize a sheep")
    parser.add_argument("-o", "--output", type=str, default="data/predictions",
                        help="where images with predictions should be saved")

    args = parser.parse_args()

    localizer = SheepLocalizer(args.model_file, args.log_file, args.gpu)
    localizer.score_threshold = args.score_threshold

    os.makedirs(args.output, exist_ok=True)

    images = args.images
    if images is None:
        images = []
        with open(args.json) as handle:
            data = json.load(handle)
            for entry in data:
                images.append(os.path.join(os.path.dirname(args.json), entry["image"]))

    for image_path in tqdm(images):
        with Image.open(image_path) as image:
            image_as_array = np.asarray(image)
            resized_image, scaling = localizer.resize(image, is_array=False)
            processed_image = localizer.preprocess(resized_image)
            bboxes, scores = localizer.localize(processed_image)

            out_image = Image.fromarray(localizer.visualize_results(image_as_array, bboxes, scores, scaling=scaling))
            out_image.save(os.path.join(args.output, os.path.basename(image_path)))
