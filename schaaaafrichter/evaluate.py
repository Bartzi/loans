import argparse
import importlib
import json
from collections import defaultdict

import chainer
import matplotlib
import os
from chainer.dataset import concat_examples
from chainercv.evaluations import eval_detection_voc

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from chainer import configuration
from chainer.backends import cuda
from chainercv.links.model.ssd.ssd_vgg16 import _imagenet_mean, SSD300, SSD512
from tqdm import tqdm

from datasets.sheep_dataset import SheepDataset
from insights.bbox_plotter import get_next_color


class Evaluator:

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.model_dir, args.log_name)) as the_log_file:
            log_data = json.load(the_log_file)[0]

        self.image_size = log_data['image_size']
        self.model_type = log_data['model_type']
        self.mean = log_data.get('image_mean', _imagenet_mean)

        self.model = self.build_model()

        if args.gpu is not None:
            self.model.to_gpu(args.gpu)

        # step 3 prepare data
        # determine whether rgb or black and white images have been used during training
        image_mode = log_data.get('image_mode', 'RGB')

        args.eval_data = args.eval_gt

        self.data_loader = SheepDataset(
            os.path.dirname(args.eval_gt),
            args.eval_data,
            image_size=self.image_size,
        )

        self.data_iterator = chainer.iterators.MultiprocessIterator(
            self.data_loader,
            args.batchsize,
            repeat=False,
            shuffle=False
        )

        self.results_path = os.path.join(self.args.model_dir, 'eval_results.json')

    def build_model(self):
        if self.model_type == 'ssd300':
            model = SSD300(n_fg_class=1)
        elif self.model_type == 'ssd512':
            model = SSD512(n_fg_class=1)
        else:
            raise NotImplementedError("Sheep Localizer is not prepared to work with model {}".format(self.model_type))

        model.score_thresh = 0.3
        return model

    def load_weights(self, snapshot_name, model):
        with np.load(os.path.join(self.args.model_dir, snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f).load(model)

    def reset(self):
        self.data_iterator.reset()

    def load_module(self, module_file):
        module_spec = importlib.util.spec_from_file_location("models.model", module_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def evaluate(self, snapshot_name=''):
        current_device = cuda.get_device_from_id(self.args.gpu)
        with current_device:
            gt_data = []
            pred_data = []

            for i, batch in enumerate(tqdm(self.data_iterator, total=len(self.data_loader) // self.args.batchsize)):
                image, gt_bboxes, gt_labels = batch[0]
                gt_data.append((gt_bboxes, gt_labels))
                # if self.args.gpu is not None:
                #     image = cuda.to_gpu(image, current_device)

                with cuda.Device(self.args.gpu):
                    with configuration.using_config('train', False):
                        bboxes, labels, scores = self.model.predict(image.copy()[None, ...])
                        if len(bboxes[0]) == 0:
                            bboxes = [np.zeros((1, 4), dtype=np.float32)]
                            labels = [np.zeros((1,), dtype=np.int32)]
                            scores = [np.zeros((1,), dtype=np.float32)]
                        pred_data.append((bboxes[0], labels[0], scores[0]))
                        # TODO handle empty predictions!!

            bboxes, labels, scores = zip(*pred_data)
            gt_bboxes, gt_labels = concat_examples(gt_data)
            result = eval_detection_voc(
                bboxes, labels, scores,
                gt_bboxes, gt_labels, None
            )
            map = result['map']

            self.save_eval_results(snapshot_name, map)

    def save_eval_results(self, snapshot_name, map):

        if os.path.exists(self.results_path):
            with open(self.results_path) as eval_file:
                json_data = json.load(eval_file)
        else:
            json_data = []

        json_data.append({
            "map": float(map),
            "snapshot_name": snapshot_name,
        })

        with open(self.results_path, 'w') as eval_file:
            json.dump(json_data, eval_file, indent=4)


def plot_eval_results(data, model_dir):
    values_per_key = defaultdict(list)

    for element in data:
        for key, value in element.items():
            values_per_key[key] += [value]

    for (key, value), color in zip(values_per_key.items(), get_next_color()):
        if key == 'snapshot_name':
            continue
        plt.plot(value, label=key)

    plt.legend()
    plt.savefig(os.path.join(model_dir, "plot.png"))

    # get max ap epoch
    best_epoch = np.argmax(np.array(values_per_key['map']))
    print(f"best ap: {max(values_per_key['map'])}")
    print(f"best epoch: {best_epoch}")
    print(f"Best Snapshot: {values_per_key['snapshot_name'][best_epoch]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluates trained localizer")
    parser.add_argument("eval_gt", help="path to gt file with all images to test")
    parser.add_argument("model_dir", help="path to directory containing train results")
    parser.add_argument("snapshot_prefix", help="prefix of snapshots to evaluate")
    parser.add_argument("--log-name", default="log", help="name of the log file [default: log]")
    parser.add_argument("--gpu", "-g", type=int, help="gpu to use [default: use cpu]")
    parser.add_argument("--num-samples", "-n", type=int, help="max number of samples to test [default: test all]")
    parser.add_argument("--batchsize", "-b", type=int, default=1, help="number of images to evaluate at once [default: 1]")

    args = parser.parse_args()

    evaluator = Evaluator(args)
    if os.path.exists(evaluator.results_path):
        # we already evaluated some snapshots, so we do not need to do that again
        with open(evaluator.results_path) as already_evaluated_model_results:
            json_data = json.load(already_evaluated_model_results)
            evaluated_snapshots = [item['snapshot_name'] for item in json_data]
    else:
        evaluated_snapshots = []

    snapshots = list(
        sorted(
            filter(
                lambda x: x not in evaluated_snapshots and args.snapshot_prefix in x,
                os.listdir(args.model_dir)
            ),
            key=lambda x: int(getattr(re.search(r"(\d+).npz", x), 'group', lambda _: 0)(1))
        )
    )
    for snapshot in tqdm(snapshots):
        try:
            evaluator.load_weights(snapshot, evaluator.model)
            evaluator.reset()
            evaluator.evaluate(snapshot)
        except Exception as e:
            print(f"Exception: {e} at snapshot: {snapshot}")

    if os.path.exists(evaluator.results_path):
        with open(evaluator.results_path) as evaluated_model_results:
            json_data = json.load(evaluated_model_results)
            plot_eval_results(json_data, args.model_dir)
