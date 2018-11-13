import argparse
import os
import json
import random

from tqdm import tqdm
from PIL import Image


class Generator:
    def __init__(self, output_path, resize_max=500, search_path=None, img_folder="images"):
        self.resize_max = resize_max
        self.test_stamps = []
        self.train_stamps = []
        self.output_path = output_path
        self.img_folder = "." if img_folder is None else img_folder
        self.search_path = search_path
        os.makedirs(os.path.join(self.output_path, self.img_folder), exist_ok=True)

        self.i = 0
        self.train_info = []
        self.test_info = []

    def load_test_stamps(self, stamps):
        self.test_stamps = []
        for stamp_path in stamps:
            self.test_stamps.append(Image.open(stamp_path))

    def load_train_stamps(self, stamps):
        self.train_stamps = []
        for stamp_path in stamps:
            self.train_stamps.append(Image.open(stamp_path))

    def get_data_for(self, image_path):
        file, _ = os.path.splitext(os.path.basename(image_path))
        directory = os.path.dirname(image_path)

        data_dir = directory
        if self.search_path is not None:
            data_dir = self.search_path

        data_file = os.path.join(data_dir, "{}.json".format(file))

        content = []
        with open(data_file) as data_handle:
            content = json.load(data_handle)
        return content

    def process_image(self, image_path, is_test):
        image = Image.open(image_path).convert(mode="RGBA")

        bounding_boxes = self.get_data_for(image_path)

        stamps = self.test_stamps if is_test else self.train_stamps

        for bounding_box in bounding_boxes:
            for stamp in stamps:
                self.make_image(image, is_test, [bounding_box], [stamp])

        for nr_bboxes in range(2, min(len(bounding_boxes), 4)):
            bboxes = random.sample(bounding_boxes, nr_bboxes)
            stamps_to_use = []
            for i in range(len(bboxes)):
                stamps_to_use.append(random.choice(stamps))
            self.make_image(image, is_test, bboxes, stamps_to_use)

    def make_image(self, image, is_test, bounding_boxes=(), stamps=()):
        if self.resize_max > 0:
            scale_factor = self.resize_max / max(image.size)

            new_size = [min(int(round(scale_factor * dim)), self.resize_max) for dim in image.size]
            image = image.resize(new_size, Image.LANCZOS)

            bounding_boxes = [[int(round(x * scale_factor)) for x in bb] for bb in bounding_boxes]

        image_output_path = self.get_next_output_path()

        target_info = self.test_info if is_test else self.train_info
        target_info.append({
            "image": image_output_path,
            # swap axis for json files
            "bounding_boxes": [[bb[1], bb[0], bb[3], bb[2]] for bb in bounding_boxes]
        })

        out = image

        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            resized_to_bb = stamps[i].resize((width, height), Image.LANCZOS)

            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            layer.paste(resized_to_bb, box=(x1, y1))
            out = Image.alpha_composite(out, layer)

        out.convert("RGB").save(os.path.join(self.output_path, image_output_path), quality=95)
        self.save_list()

    def get_next_output_path(self):
        self.i += 1
        return os.path.join(self.img_folder, "{:06d}.jpg".format(self.i - 1))

    def save_list(self):
        with open(os.path.join(self.output_path, "train_info.json"), "w") as list_file:
            list_file.write(json.dumps(self.train_info, indent=2))

        with open(os.path.join(self.output_path, "test_info.json"), "w") as list_file:
            list_file.write(json.dumps(self.test_info, indent=2))


def main(args):
    prev_state = random.getstate()
    random.seed(42)

    images = [os.path.join(args.image_folder, i) for i in sorted(os.listdir(args.image_folder))]

    nr_test_images = int(args.split * len(images))
    is_test = [True] * nr_test_images + [False] * (len(images) - nr_test_images)
    random.shuffle(is_test)

    generator = Generator(args.output_path, args.resize_max, args.search_path)
    generator.load_test_stamps(args.test_stamps)
    generator.load_train_stamps(args.train_stamps)

    for i, image_path in enumerate(tqdm(images)):
        generator.process_image(image_path, is_test[i])

    random.setstate(prev_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Paste a number of images into other images with bounding boxes",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image-folder", required=True, help="folder with template image files")
    parser.add_argument("--test-stamps", required=True, nargs="+", help="path to search for images to paste")
    parser.add_argument("--train-stamps", required=True, nargs="+", help="path to search for images to paste")
    parser.add_argument("--search-path", default=None, help="path to search for corresponding json files")
    parser.add_argument("--output-path", default="data/generated", help="output path directory")
    parser.add_argument("--ext", default="jpg", help="extension of image files")
    parser.add_argument("--split", default=0.2, type=float, help="define percentage of images in test data")
    parser.add_argument("--resize-max", default=500, type=int, help="resize the larger image axis to this value (keeps "
                                                                    "aspect ratio), set to <= 0 to disable resizing")

    main(parser.parse_args())
