import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, initializers
from chainer.backends import cuda
from chainer.links.model.vision import resnet
from chainer.links.model.vision.resnet import _global_average_pooling_2d
from chainercv.links.model.resnet import ResBlock

from common.utils import Size
from functions.rotation_droput import rotation_dropout
from insights.visual_backprop import VisualBackprop
from iou.iou_regressor import MyResNet50Layers
from sheep.resnet import ResNet, BasicBlock


class SheepLocalizer(Chain):

    def __init__(self, out_size, transform_rois_to_grayscale=False, train_imagenet=False):
        super().__init__()
        with self.init_scope():
            self.feature_extractor = ResNet(18, class_labels=1000 if train_imagenet else None)

            if not train_imagenet:
                self.res6 = BasicBlock(2, 512)
                self.res7 = BasicBlock(2, 512)
                self.param_predictor = L.Linear(512, 6)

                transform_bias = self.param_predictor.b.data
                transform_bias[[0, 4]] = 0.8
                transform_bias[[2, 5]] = 0
                self.param_predictor.W.data[...] = 0

        self.visual_backprop_anchors = []
        self.out_size = out_size
        self.transform_rois_to_grayscale = transform_rois_to_grayscale
        self.visual_backprop = VisualBackprop()
        self.train_imagenet = train_imagenet

    def __call__(self, images):
        self.visual_backprop_anchors.clear()

        with cuda.Device(images.data.device):
            input_images = self.prepare_images(images.copy() * 255)
        h = self.feature_extractor(input_images)

        if self.train_imagenet:
            return h

        if images.shape[-2] > 224:
            h = self.res6(h)

            if images.shape[-2] > 300:
                h = self.res7(h)

        self.visual_backprop_anchors.append(h)
        h = _global_average_pooling_2d(h)

        transform_params = self.param_predictor(h)
        transform_params = rotation_dropout(F.reshape(transform_params, (-1, 2, 3)), ratio=0.0)
        points = F.spatial_transformer_grid(transform_params, self.out_size)
        rois = F.spatial_transformer_sampler(images, points)

        if self.transform_rois_to_grayscale:
            assert rois.shape[1] == 3, "rois are not in RGB, can not convert them to grayscale"
            b, g, r = F.split_axis(rois, 3, axis=1)
            rois = 0.299 * r + 0.587 * g + 0.114 * b

        return rois, points

    def prepare_images(self, images):
        if self.xp != np:
            device = images.data.device
            images = F.copy(images, -1)

        converted_images = [resnet.prepare(image.data, size=None) for image in F.separate(images, axis=0)]
        converted_images = F.stack(converted_images, axis=0)

        if self.xp != np:
            converted_images = F.copy(converted_images, device.id)
        return converted_images

    def extract_corners(self, bboxes):
        top = bboxes[:, 1, 0, 0]
        left = bboxes[:, 0, 0, 0]
        bottom = bboxes[:, 1, -1, -1]
        right = bboxes[:, 0, -1, -1]

        corners = F.stack([top, left, bottom, right], axis=1)
        return corners

    def scale_bboxes(self, bboxes, image_size):
        bboxes = (bboxes + 1) / 2
        bboxes.data[:, ::2] *= image_size.height
        bboxes.data[:, 1::2] *= image_size.width
        return bboxes

    def predict(self, images, return_visual_backprop=False):
        with cuda.Device(self._device_id):
            images = [self.xp.array(image) for image in images]
            images = self.xp.stack(images, axis=0)
            with chainer.using_config('train', False):
                rois, bboxes = self(images)
                if return_visual_backprop:
                    if not hasattr(self, 'visual_backprop'):
                        self.visual_backprop = VisualBackprop()
                    visual_backprop = cuda.to_cpu(self.visual_backprop.perform_visual_backprop(self.visual_backprop_anchors[0]))
                else:
                    visual_backprop = None

                bboxes = self.extract_corners(bboxes)
                bboxes = self.scale_bboxes(bboxes, Size._make(images.shape[-2:]))

        bboxes = [cuda.to_cpu(bbox).reshape(1, -1) for bbox in bboxes.data]

        return bboxes, rois, np.ones((len(bboxes), 1)), visual_backprop


class Resnet50SheepLocalizer(SheepLocalizer):

    def __init__(self, out_size, transform_rois_to_grayscale=False, train_imagenet=False):
        super(SheepLocalizer, self).__init__()
        initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {"initialW": initialW}
        keys_to_remove = ['fc6', 'prob'] if not train_imagenet else ['prob']
        with self.init_scope():
            self.feature_extractor = MyResNet50Layers(keys_to_remove=keys_to_remove, pretrained_model='auto')
            if not train_imagenet:
                self.param_predictor = L.Linear(2048, 6)

                self.res6 = ResBlock(2, None, 1024, 2048, 2, **kwargs)
                self.res7 = ResBlock(2, None, 1024, 2048, 2, **kwargs)

                transform_bias = self.param_predictor.b.data
                transform_bias[[0, 4]] = 0.8
                transform_bias[[2, 5]] = 0
                self.param_predictor.W.data[...] = 0

        self.visual_backprop_anchors = []
        self.out_size = out_size
        self.transform_rois_to_grayscale = transform_rois_to_grayscale
        self.train_imagenet = train_imagenet

    def __call__(self, images):
        self.visual_backprop_anchors.clear()

        with cuda.Device(images.data.device):
            input_images = self.prepare_images(images.copy() * 255)

        if self.train_imagenet:
            return self.feature_extractor(input_images, layers=['fc6'])['fc6']
        else:
            h = self.feature_extractor(input_images, layers=['res5', 'pool5'])

        self.visual_backprop_anchors.append(h['res5'])
        if images.shape[-2] > 224:
            h = h['res5']
            h = self.res6(h)

            if images.shape[-2] > 300:
                h = self.res7(h)

            h = _global_average_pooling_2d(h)
        else:
            h = h['pool5']

        transform_params = self.param_predictor(h)
        transform_params = rotation_dropout(F.reshape(transform_params, (-1, 2, 3)), ratio=0.0)
        points = F.spatial_transformer_grid(transform_params, self.out_size)
        rois = F.spatial_transformer_sampler(images, points)

        if self.transform_rois_to_grayscale:
            assert rois.shape[1] == 3, "rois are not in RGB, can not convert them to grayscale"
            b, g, r = F.split_axis(rois, 3, axis=1)
            rois = 0.299 * r + 0.587 * g + 0.114 * b

        return rois, points
