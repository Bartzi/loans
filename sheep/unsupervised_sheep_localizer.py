import chainer
import numpy as np
import os

from schaaaafrichter.sheeping.sheep_localizer import SheepLocalizer
from train_utils.module_loading import get_class


class UnsupervisedSheepLocalizer(SheepLocalizer):

    def __init__(self, *args, **kwargs):
        self.discriminator_model_file = kwargs.pop('discriminator', None)
        self.discriminator = None
        super().__init__(*args, **kwargs)
        self.target_size = self.log.get('target_size', [75, 75])

    def build_model(self):
        localizer_class = get_class(*self.log['localizer'], os.path.dirname(self.model_file))
        model = localizer_class(self.target_size)

        self.model = self.init_model(model, self.model_file)
        self.initialized = True

        if self.discriminator_model_file is not None:
            discriminator_class = get_class(*self.log['discriminator'], os.path.dirname(self.discriminator_model_file))
            discriminator = discriminator_class(self.log['discriminator_output_dim'])

            self.discriminator = self.init_model(discriminator, self.discriminator_model_file)

    def init_model(self, model_object, model_weights):
        if self.gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(self.gpu_id).use()
            model_object.to_gpu()

        with np.load(model_weights) as f:
            chainer.serializers.NpzDeserializer(f).load(model_object)

        return model_object

    def localize(self, processed_image, return_visual_backprop=False):
        if not self.initialized:
            self.build_model()
        bboxes, rois, scores, visual_backprop = self.model.predict([processed_image], return_visual_backprop=return_visual_backprop)

        if self.discriminator is not None:
            scores = self.discriminator(rois).data
            if float(scores) < self.score_threshold:
                return np.zeros((1, 1)), np.zeros((1, 1)), None

        if return_visual_backprop:
            visual_backprop = np.tile(visual_backprop, [3, 1, 1])
            visual_backprop = np.transpose(visual_backprop, (0, 2, 3, 1))
            visual_backprop = np.ascontiguousarray((visual_backprop * 255).astype(np.uint8))
            return bboxes[0], scores[0], visual_backprop[0]

        return bboxes[0], scores[0], None

    def preprocess(self, image, make_copy=True, bgr_to_rgb=False):
        if make_copy:
            image = image.copy()

        if bgr_to_rgb:
            b, g, r = np.split(image, 3, axis=2)
            image = np.concatenate((r, g, b), axis=2)

        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        return image / 255
