import os

import argparse
import copy
import numpy as np

import chainer
from chainer.datasets import ConcatenatedDataset, split_dataset_n_random, split_dataset
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

from datasets.sheep_dataset import SheepDataset
from insights.bbox_plotter import BBOXPlotter
from iterators.thread_iterator import ThreadIterator


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img,
                max_ratio=2,
                fill=self.mean,
                return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img,
            bbox,
            return_param=True,
        )
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="path to train json file")
    parser.add_argument('test_dataset', help="path to test dataset json file")
    parser.add_argument('--dataset-root', help="path to dataset root if dataset file is not already in root folder of dataset")
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd512')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpu', type=int, nargs='*', default=[])
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    parser.add_argument('--lr', type=float, default=0.001, help="default learning rate")
    parser.add_argument('--port', type=int, default=1337, help="port for bbox sending")
    parser.add_argument('--ip', default='127.0.0.1', help="destination ip for bbox sending")
    parser.add_argument('--test-image', help="path to test image that shall be displayed in bbox vis")
    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.dirname(args.dataset)

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=1,
            pretrained_model='imagenet')
        image_size = (300, 300)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=1,
            pretrained_model='imagenet')
        image_size = (512, 512)
    else:
        raise NotImplementedError("The model you want to train does not exist")

    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    train = TransformDataset(
        SheepDataset(args.dataset_root, args.dataset, image_size=image_size),
        Transform(model.coder, model.insize, model.mean)
    )

    if len(args.gpu) > 1:
        gpu_datasets = split_dataset_n_random(train, len(args.gpu))
        if not len(gpu_datasets[0]) == len(gpu_datasets[-1]):
            adapted_second_split = split_dataset(gpu_datasets[-1], len(gpu_datasets[0]))[0]
            gpu_datasets[-1] = adapted_second_split
    else:
        gpu_datasets = [train]

    train_iter = [ThreadIterator(gpu_dataset, args.batchsize) for gpu_dataset in gpu_datasets]

    test = SheepDataset(args.dataset_root, args.test_dataset, image_size=image_size)
    test_iter = chainer.iterators.MultithreadIterator(
        test, args.batchsize, repeat=False, shuffle=False, n_threads=2)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    if len(args.gpu) <= 1:
        updater = training.updaters.StandardUpdater(
            train_iter[0],
            optimizer,
            device=args.gpu[0] if len(args.gpu) > 0 else -1,
        )
    else:
        updater = training.updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=args.gpu)
        updater.setup_workers()

    if len(args.gpu) > 0 and args.gpu[0] >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu[0]).use()
        model.to_gpu()

    trainer = training.Trainer(updater, (200, 'epoch'), args.out)

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(1000, 'iteration'))

    # build logger
    # make sure to log all data necessary for prediction
    log_interval = 100, 'iteration'
    data_to_log = {
        'image_size': image_size,
        'model_type': args.model,
    }

    # add all command line arguments
    for argument in filter(lambda x: not x.startswith('_'), dir(args)):
        data_to_log[argument] = getattr(args, argument)

    # create callback that logs all auxiliary data the first time things get logged
    def backup_train_config(stats_cpu):
        if stats_cpu['iteration'] == log_interval:
            stats_cpu.update(data_to_log)

    trainer.extend(extensions.LogReport(trigger=log_interval, postprocess=backup_train_config))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/loc', 'main/loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(5000, 'iteration'))

    if args.test_image is not None:
        plot_image = train._dataset.load_image(args.test_image, resize_to=image_size)
    else:
        plot_image, _, _ = train.get_example(0)
        plot_image += train._transform.mean

    bbox_plotter = BBOXPlotter(
        plot_image,
        os.path.join(args.out, 'bboxes'),
        send_bboxes=True,
        upstream_port=args.port,
        upstream_ip=args.ip,
    )
    trainer.extend(bbox_plotter, trigger=(10, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
