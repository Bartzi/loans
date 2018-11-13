import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import Chain, initializers
from chainer.links import ResNet50Layers
from chainer.links.model.vision import resnet
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import ResBlock

from common.utils import Size
from detection_dcgan.lstm_rpn_net import RegionProposalNetwork
from functions.rotation_droput import rotation_dropout


class IOURegressor(Chain):

    def __init__(self):
        super().__init__()
        initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {"initialW": initialW}
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=False)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=2, stride=2)

            self.res1 = ResBlock(2, None, 64, 64, 1, **kwargs)
            self.res2 = ResBlock(2, None, 128, 128, 2, **kwargs)
            self.res3 = ResBlock(2, None, 256, 256, 2, **kwargs)
            self.res4 = ResBlock(2, None, 512, 512, 2, **kwargs)

            self.regressor = L.Linear(None, 1, initialW=initializers.Normal(scale=0.01))

    def __call__(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)

        return F.flatten(F.sigmoid(self.regressor(h)))


class IOULocalizer(Chain):

    def __init__(self, out_size, rpn):
        super().__init__()
        initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        kwargs = {"initialW": initialW}
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=False)

            self.res1 = ResBlock(2, None, 64, 64, 1, **kwargs)
            self.res2 = ResBlock(2, None, 128, 128, 2, **kwargs)
            self.res3 = ResBlock(2, None, 256, 256, 2, **kwargs)
            self.res4 = ResBlock(2, None, 512, 512, 2, **kwargs)

            self.rpn = rpn

        self.visual_backprop_anchors = []
        self.out_size = out_size

    def __call__(self, images):
        self.visual_backprop_anchors.clear()

        h = self.conv1(images)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)

        self.visual_backprop_anchors.append(h)

        objectness_scores, points, rois = self.extract_rois(h, images)
        return rois, points, objectness_scores

    def extract_rois(self, h, images):
        batch_size = len(h)
        image_size = Size._make(images.shape[-2:])
        objectness_scores, points, rois = self.rpn(h, image_size, self.out_size, images)
        points = F.reshape(points, (batch_size, -1) + points.shape[1:])
        points = F.reshape(points, (-1,) + points.shape[2:])
        objectness_scores = F.reshape(objectness_scores, (-1,) + objectness_scores.shape[2:])
        return objectness_scores, points, rois


class MyResNet50Layers(ResNet50Layers):

    def __init__(self, *args, **kwargs):
        self.keys_to_remove = kwargs.pop('keys_to_remove', [])
        super().__init__(*args, **kwargs)

    @property
    def functions(self):
        funcs = super().functions
        for key in self.keys_to_remove:
            del funcs[key]
        return funcs


class Resnet50IOULocalizer(IOULocalizer):

    def __init__(self, out_size):
        super(IOULocalizer, self).__init__()
        with self.init_scope():
            self.resnet = MyResNet50Layers()
            self.rpn = RegionProposalNetwork()

        self.visual_backprop_anchors = []
        self.out_size = out_size

    def __call__(self, images):
        self.visual_backprop_anchors.clear()
        converted_images = self.prepare_images(images)
        h = self.resnet(converted_images, layers=['res5'])['res5']

        self.visual_backprop_anchors.append(h)

        objectness_scores, points, rois = self.extract_rois(h, images)
        return rois, points, objectness_scores

    def prepare_images(self, images):
        if self.xp != np:
            device = images.data.device
            images = F.copy(images, -1)

        converted_images = [resnet.prepare(image.data, size=None) for image in F.separate(images, axis=0)]
        converted_images = F.stack(converted_images, axis=0)

        if self.xp != np:
            converted_images = F.copy(converted_images, device.id)
        return converted_images
