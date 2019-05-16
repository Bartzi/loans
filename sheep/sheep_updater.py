import chainer
import chainer.functions as F
from chainer.backends import cuda

from common.utils import DirectionLossCalculator, OutOfImageLossCalculator, Size


class SheepAssessor(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.localizer, self.discriminator = kwargs.pop('models')
        self.anchor_iter = kwargs.pop('anchor_iter', None)
        self.create_pca = kwargs.pop('create_pca', False)
        self.n_components_pca = kwargs.pop('n_components_pca', 2)
        self.pca = None
        self.freeze_discriminator = kwargs.pop('resume_discriminator', None) is not None
        self.localizer_target = kwargs.pop('localizer_target', 1.0)

        super().__init__(*args, **kwargs)

        self.regularizers = [
            DirectionLossCalculator(self.localizer.xp),
            OutOfImageLossCalculator(self.localizer.xp)
        ]

    def update_core(self):
        localizer_optimizer = self.get_optimizer('opt_gen')
        discriminator_optimizer = self.get_optimizer('opt_dis')
        xp = self.localizer.xp

        with cuda.Device(self.device):
            batch = next(self.get_iterator('real'))
            real_images, labels = self.converter(batch, self.device)[:2]

            y_real = self.discriminator(real_images)

            batch = next(self.get_iterator('main'))
            fake_images = self.converter(batch, self.device)
            x_fake, bboxes = self.localizer(fake_images)
            y_fake = self.discriminator(x_fake)

            localization_labels = xp.full((len(y_fake), 1), self.localizer_target, dtype=xp.float32)
            loss_localizer = F.mean_squared_error(y_fake, localization_labels)

            for regularizer in self.regularizers:
                loss_localizer += regularizer.calc_loss(bboxes, Size._make(fake_images.shape[-2:]))

            self.discriminator.disable_update()

            self.localizer.cleargrads()
            loss_localizer.backward()
            localizer_optimizer.update()
            chainer.reporter.report({'loss_localizer': loss_localizer})

            self.discriminator.enable_update()

            x_fake.unchain_backward()
            bboxes.unchain_backward()

            loss_dis = F.mean_squared_error(y_real, labels)

            if not self.freeze_discriminator:
                self.discriminator.cleargrads()
                self.localizer.cleargrads()
                loss_dis.backward()
                discriminator_optimizer.update()

            chainer.reporter.report({'loss_dis': loss_dis})
