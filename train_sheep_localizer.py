import argparse
import datetime
import json
import os

import chainer
import numpy as np
from PIL import Image
from chainer.datasets import get_mnist
from chainer.training import extensions
from chainer.training.extensions import Evaluator

from commands.interactive_train import open_interactive_prompt
from common.datasets.image_dataset import ImageDataset, LabeledImageDataset
from common.net import ResnetAssessor
from insights.bbox_plotter import BBOXPlotter
from sheep.sheep_evaluator import SheepMAPEvaluator
from sheep.sheep_localizer import Resnet50SheepLocalizer, SheepLocalizer
from sheep.sheep_updater import SheepAssessor
from train_utils.logger import Logger
from train_utils.train_utils import get_definition_filepath


def load_train_paths(train_file, with_label=False):
    with open(train_file) as handle:
        train_data = json.load(handle)

    paths = [item["image"] for item in train_data]
    if with_label:
        labels = [item['bounding_boxes'][0] for item in train_data]
        return list(zip(paths, labels))
    return paths


def load_image(image_path, size):
    with Image.open(image_path) as anchor_image:
        anchor_image = anchor_image.convert('RGB')
        anchor_image = anchor_image.resize(size)
        anchor_image = np.array(anchor_image).astype(np.float32)
        anchor_image = np.transpose(anchor_image, (2, 0, 1))
        anchor_image /= 255
    return anchor_image


def load_pretrained_model(model_file, model):
    with np.load(model_file) as handle:
        chainer.serializers.NpzDeserializer(handle, strict=False).load(model)


def main():
    parser = argparse.ArgumentParser(description="Train a sheep localizer")
    parser.add_argument("train_file", help="path to train csv")
    parser.add_argument("val_file", help="path to validation file (if you do not want to do validation just enter gibberish here")
    parser.add_argument("reference_file", help="path to reference images with different zoom levels")
    parser.add_argument("--no-validation", dest='validation', action='store_false', default=True, help="don't do validation")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), help="input size for localizer")
    parser.add_argument("--target-size", type=int, nargs=2, default=(75, 75), help="crop size for each image")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size for training")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="gpu if to use (-1 means cpu)")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-l", "--log-dir", default='sheep_logs', help="path to log dir")
    parser.add_argument("--ln", "--log-name", default="test", help="name of log")
    parser.add_argument("--num-epoch", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--snapshot-interval", type=int, default=1000, help="number of iterations after which a snapshot will be taken")
    parser.add_argument("--no-snapshot-every-epoch", dest="snapshot_every_epoch", action='store_false', default=True, help="Do not take a snapshot on every epoch")
    parser.add_argument("--log-interval", type=int, default=100, help="log interval")
    parser.add_argument("--port", type=int, default=1337, help="port that is used by bbox plotter to send predictions on test image")
    parser.add_argument("--test-image", help="path to test image that is to be used with bbox plotter")
    parser.add_argument("--anchor-image", help="path to anchor image used for metric learning")
    parser.add_argument("--rl", dest="resume_localizer", help="path to snapshot that is to be used to resume training of localizer")
    parser.add_argument("--rd", dest="resume_discriminator", help="path to snapshot that is to be used to pre-initialize discriminator")
    parser.add_argument("--use-resnet-18", action='store_true', default=False, help="Use Resnet-18 for localization")
    parser.add_argument("--localizer-target", type=float, default=1.0, help="target iou for localizer to reach in the interval [0,1]")
    parser.add_argument("--no-imgaug", action='store_false', dest='use_imgaug', default=True, help="disable image augmentation with `imgaug`, but use naive image augmentation instead")

    args = parser.parse_args()

    report_keys = ["epoch", "iteration", "loss_localizer", "loss_dis", "map", "mean_iou"]

    if args.train_file.endswith('.json'):
        train_image_paths = load_train_paths(args.train_file)
    else:
        train_image_paths = args.train_file

    train_dataset = ImageDataset(
        train_image_paths,
        os.path.dirname(args.train_file),
        image_size=args.image_size,
        dtype=np.float32,
        use_imgaug=args.use_imgaug,
        transform_probability=0.5,
    )

    if args.reference_file == 'mnist':
        reference_dataset = get_mnist(withlabel=False, ndim=3, rgb_format=True)[0]
        args.target_size = (28, 28)
    else:
        reference_dataset = LabeledImageDataset(
            args.reference_file,
            os.path.dirname(args.reference_file),
            image_size=args.target_size,
            dtype=np.float32,
            label_dtype=np.float32,
        )

    if args.validation:
        if args.val_file.endswith('.json'):
            validation_data = load_train_paths(args.val_file, with_label=True)
        else:
            validation_data = args.val_file

        validation_dataset = LabeledImageDataset(validation_data, os.path.dirname(args.val_file), image_size=args.image_size)
        validation_iter = chainer.iterators.MultithreadIterator(validation_dataset, args.batch_size, repeat=False)

    data_iter = chainer.iterators.MultithreadIterator(train_dataset, args.batch_size)
    reference_iter = chainer.iterators.MultithreadIterator(reference_dataset, args.batch_size)

    localizer_class = SheepLocalizer if args.use_resnet_18 else Resnet50SheepLocalizer
    localizer = localizer_class(args.target_size)

    if args.resume_localizer is not None:
        load_pretrained_model(args.resume_localizer, localizer)

    discriminator_output_dim = 1
    discriminator = ResnetAssessor(output_dim=discriminator_output_dim)
    if args.resume_discriminator is not None:
        load_pretrained_model(args.resume_discriminator, discriminator)
    models = [localizer, discriminator]

    localizer_optimizer = chainer.optimizers.Adam(alpha=args.learning_rate, amsgrad=True)
    localizer_optimizer.setup(localizer)

    discriminator_optimizer = chainer.optimizers.Adam(alpha=args.learning_rate, amsgrad=True)
    discriminator_optimizer.setup(discriminator)

    optimizers = [localizer_optimizer, discriminator_optimizer]

    updater_args = {
        "iterator": {
            'main': data_iter,
            'real': reference_iter,
        },
        "device": args.gpu,
        "optimizer": {
            "opt_gen": localizer_optimizer,
            "opt_dis": discriminator_optimizer,
        },
        "create_pca": False,
        "resume_discriminator": args.resume_discriminator,
        "localizer_target": args.localizer_target,
    }

    updater = SheepAssessor(
        models=[localizer, discriminator],
        **updater_args
    )

    log_dir = os.path.join(args.log_dir, "{}_{}".format(datetime.datetime.now().isoformat(), args.ln))
    args.log_dir = log_dir
    # create log dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    trainer = chainer.training.Trainer(updater, (args.num_epoch, 'epoch'), out=args.log_dir)

    data_to_log = {
        'log_dir': args.log_dir,
        'image_size': args.image_size,
        'updater': [updater.__class__.__name__, 'updater.py'],
        'discriminator': [discriminator.__class__.__name__, 'discriminator.py'],
        'discriminator_output_dim': discriminator_output_dim,
        'localizer': [localizer.__class__.__name__, 'localizer.py']
    }

    for argument in filter(lambda x: not x.startswith('_'), dir(args)):
        data_to_log[argument] = getattr(args, argument)

    def backup_train_config(stats_cpu):
        if stats_cpu['iteration'] == args.log_interval:
            stats_cpu.update(data_to_log)

    for model in models:
        trainer.extend(
            extensions.snapshot_object(model, model.__class__.__name__ + '_{.updater.iteration}.npz'),
            trigger=lambda trainer: trainer.updater.is_new_epoch if args.snapshot_every_epoch else trainer.updater.iteration % args.snapshot_interval == 0,
        )

    # log train information everytime we encouter a new epoch or args.log_interval iterations have been done
    log_interval_trigger = (lambda trainer:
                            trainer.updater.is_new_epoch or trainer.updater.iteration % args.log_interval == 0)

    sheep_evaluator = SheepMAPEvaluator(localizer, args.gpu)
    if args.validation:
        trainer.extend(
            Evaluator(validation_iter, localizer, device=args.gpu, eval_func=sheep_evaluator),
            trigger=log_interval_trigger,
        )

    models.append(updater)
    logger = Logger(
        [get_definition_filepath(model) for model in models],
        args.log_dir,
        postprocess=backup_train_config,
        trigger=log_interval_trigger,
        dest_file_names=['localizer.py', 'discriminator.py', 'updater.py'],
    )

    if args.test_image is not None:
        plot_image = load_image(args.test_image, args.image_size)
        gt_bbox = None
    else:
        if args.validation:
            plot_image, gt_bbox, _ = validation_dataset.get_example(0)
        else:
            plot_image = train_dataset.get_example(0)
            gt_bbox = None

    bbox_plotter = BBOXPlotter(
        plot_image,
        os.path.join(args.log_dir, 'bboxes'),
        args.target_size,
        send_bboxes=True,
        upstream_port=args.port,
        visualization_anchors=[
            ["visual_backprop_anchors"],
        ],
        device=args.gpu,
        render_extracted_rois=True,
        num_rois_to_render=4,
        show_visual_backprop_overlay=False,
        show_backprop_and_feature_vis=True,
        gt_bbox=gt_bbox,
        render_pca=True,
        log_name=args.ln,
    )
    trainer.extend(bbox_plotter, trigger=(1, 'iteration'))

    trainer.extend(
        logger,
        trigger=log_interval_trigger
    )
    trainer.extend(
        extensions.PrintReport(report_keys, log_report='Logger'),
        trigger=log_interval_trigger
    )

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.dump_graph('loss_localizer', out_name='model.dot'))

    open_interactive_prompt(
        bbox_plotter=bbox_plotter,
        optimizer=optimizers,
    )

    trainer.run()


if __name__ == "__main__":
    main()
