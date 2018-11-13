import json
import os

import numpy as np

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from PIL import Image


class SheepDataset(GetterDataset):

    def __init__(self, dataset_root, label_file, image_size=(512, 512)):
        super().__init__()

        self.image_size = image_size
        self.dataset_root = dataset_root
        with open(label_file) as the_label_file:
            self.data = json.load(the_label_file)

        self.add_getter('img', self.get_image)
        self.add_getter(('bbox', 'label'), self.get_annotation)

        self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path, resize_to=None):
        with Image.open(image_path) as image:
            if resize_to is not None:
                image = image.resize(resize_to)
            image = image.convert('RGB')
            img = np.array(image, dtype=np.float32)

        return img.transpose((2, 0, 1))

    def get_image(self, i):
        image_path = os.path.join(self.dataset_root, self.data[i]['image'])
        return self.load_image(image_path)

    def get_annotation(self, i):
        bboxes = self.data[i]['bounding_boxes']

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.zeros((len(bboxes)), dtype=np.int32)

        return bboxes, labels
