import csv
import random

import numpy
import six

from PIL import Image

from chainer.datasets import ImageDataset as ChainerImageDataset
from chainer.datasets import LabeledImageDataset as ChainerLabeledImageDataset
from chainercv import transforms
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def resize_image(image, image_size, image_mode='RGB'):
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image.astype('uint8'))
    else:
        pil_image = Image.fromarray(image.transpose(1, 2, 0).astype('uint8'))
    pil_image = pil_image.convert(image_mode)
    pil_image = pil_image.resize((image_size[1], image_size[0]), Image.LANCZOS)

    if image_mode == 'L':
        image = numpy.asarray(pil_image).astype(numpy.float32)
    else:
        image = numpy.asarray(pil_image).transpose(2, 0, 1).astype(numpy.float32)
    return image


def rotate_image(image, min_angle, max_angle, image_mode='RGB'):
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image.astype('uint8'))
    else:
        pil_image = Image.fromarray(image.transpose(1, 2, 0).astype('uint8'))
    pil_image = pil_image.convert(image_mode)
    rotation_angle = random.randint(min_angle, max_angle)
    pil_image = pil_image.rotate(rotation_angle, expand=False)

    if image_mode == 'L':
        image = numpy.asarray(pil_image).astype(numpy.float32)
    else:
        image = numpy.asarray(pil_image).transpose(2, 0, 1).astype(numpy.float32)
    return image


class ImageDataset(ChainerImageDataset):

    def __init__(self, *args, **kwargs):
        self.image_size = kwargs.pop('image_size', None)
        self.image_mode = kwargs.pop('image_mode', 'RGB')
        self.transform_probability = kwargs.pop('transform_probability', 0)
        self.use_imgaug = kwargs.pop('use_imgaug', True)
        self.min_crop_ratio = kwargs.pop('min_crop_ratio', 0.6)
        self.max_crop_ratio = kwargs.pop('max_crop_ratio', 0.9)
        self.crop_always = kwargs.pop('crop_always', False)
        if self.transform_probability > 0 and self.use_imgaug:
            self.augmentations = iaa.Sometimes(
                self.transform_probability,
                iaa.SomeOf(
                    (0, None),
                    [
                        iaa.Fliplr(1.0),
                        iaa.AddToHueAndSaturation(iap.Uniform(-20, 20), per_channel=True),
                        iaa.CropAndPad(percent=(-0.10, 0.10), pad_mode=["constant", "edge"]),
                    ],
                    random_order=True
                )
            )
        else:
            self.augmentations = None

        super().__init__(*args, **kwargs)

    def get_example(self, i):
        image = super().get_example(i)
        if image.shape[0] == 1:
            image = numpy.tile(image, (3, 1, 1))

        if self.augmentations is not None and self.use_imgaug:
            image = numpy.transpose(image, (1, 2, 0))
            image = image.astype(numpy.uint8)
            image = self.augmentations.augment_images([image])[0]
            image = image.astype(numpy.float32)
            image = numpy.transpose(image, (2, 0, 1))
        elif random.random() < self.transform_probability:
            if self.crop_always or random.random() <= 0.5:
                crop_ratio = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
                image = transforms.random_crop(image, tuple([int(size * crop_ratio) for size in image.shape[-2:]]))
            image = transforms.random_flip(image, x_random=True)

        if self.image_size is not None:
            image = resize_image(image, self.image_size, image_mode=self.image_mode)

        if len(image.shape) == 2:
            image = image[None, ...]

        return image / 255


class LabeledImageDataset(ChainerLabeledImageDataset):

    def __init__(self, pairs, root='.', dtype=numpy.float32, label_dtype=numpy.int32, image_size=None, image_mode='RGB', transform_probability=0, return_dummy_scores=True):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                reader = csv.reader(pairs_file, delimiter='\t')
                for i, pair in enumerate(reader):
                    pairs.append((pair[0], list(map(label_dtype, pair[1:]))))
        self.transform_probability = transform_probability
        if self.transform_probability > 0:
            self.augmentations = iaa.Sometimes(
                self.transform_probability,
                iaa.SomeOf(
                    (0, None),
                    [
                        iaa.Fliplr(1.0),
                        iaa.AddToHueAndSaturation(iap.Uniform(-20, 20), per_channel=True),
                        iaa.ContrastNormalization((0.75, 1.0)),
                        iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    ],
                    random_order=True
                )
            )
        else:
            self.augmentations = None
        self._pairs = pairs
        self._root = root
        self._dtype = dtype
        self._label_dtype = label_dtype
        self.image_size = image_size
        self.image_mode = image_mode
        self.return_dummy_scores = return_dummy_scores

    def shrink_dataset(self, new_size):
        self._pairs = self._pairs[:new_size]

    def check_for_bad_label(self, label, image_size):
        error_text = f"Label can not be scaled correctly are you sure you created the dataset correctly, and provided the correct sizes? Image size: {image_size}, label: {label}"
        ten_percent_extra = [size * 0.1 for size in image_size]
        assert (label[:, 0] >= 0 - ten_percent_extra[0]).all(), error_text
        assert (label[:, 1] >= 0 - ten_percent_extra[1]).all(), error_text
        assert (label[:, 2] <= image_size[0] + ten_percent_extra[0]).all(), error_text
        assert (label[:, 3] <= image_size[1] + ten_percent_extra[1]).all(), error_text

    def get_example(self, i):
        try:
            image, label = super().get_example(i)
        except Exception as e:
            print(e)
            image, label = super().get_example(0)

        if len(label.shape) > 0 and len(label) % 4 == 0:
            num_bboxes = len(label) // 4
            label = numpy.reshape(label, (num_bboxes, -1))

        if image.shape[0] == 1:
            image = numpy.tile(image, (3, 1, 1))

        if self.augmentations is not None:
            image = numpy.transpose(image, (1, 2, 0))
            image = image.astype(numpy.uint8)
            image = self.augmentations.augment_images([image])[0]
            image = image.astype(numpy.float32)
            image = numpy.transpose(image, (2, 0, 1))

        if self.image_size is not None:
            image_size = image.shape[-2:]
            if len(label.shape) > 1:
                # we are likely dealing with bboxes
                self.check_for_bad_label(label, image_size)
                label = transforms.resize_bbox(label.astype(numpy.float32), image_size, self.image_size)
            image = resize_image(image, self.image_size, image_mode=self.image_mode)
            label = label.astype(self._label_dtype)

        if len(image.shape) == 2:
            image = image[None, ...]

        if self.return_dummy_scores:
            return image / 255, label, numpy.zeros((1,))
        return image / 255, label


class DiscriminatorImageDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop('label')
        super().__init__(*args, **kwargs)

    def get_example(self, i):
        image = super().get_example(i)
        return image, numpy.array([self.label])
