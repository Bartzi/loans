import argparse
import importlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import chainer
import chainer.functions as F
import chainercv
import matplotlib
from PIL import ImageDraw
from chainer.dataset import concat_examples

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from chainer import configuration
from chainer.backends import cuda
from chainercv.utils import non_maximum_suppression, bbox_iou
from tqdm import tqdm
from xml.dom import minidom

from common.datasets.image_dataset import LabeledImageDataset
from insights.bbox_plotter import BBOXPlotter, get_next_color
from train_sheep_localizer import load_train_paths
from train_utils.datatypes import Size
from train_utils.match_bbox import get_aabb_corners
from train_utils.module_loading import get_class


class Evaluator:

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.model_dir, args.log_name)) as the_log_file:
            log_data = json.load(the_log_file)[0]

        self.image_size = log_data['image_size']
        self.target_size = log_data['target_size']

        # step 1 build network
        localizer_class = get_class(*log_data['localizer'], args.model_dir)
        self.localizer = localizer_class(self.target_size)

        if args.assessor is not None:
            discriminator_class = get_class(*log_data['discriminator'], args.model_dir)
            self.discriminator = discriminator_class()
            self.load_weights(args.assessor, self.discriminator)
        else:
            self.discriminator = None

        if args.gpu is not None:
            self.localizer.to_gpu(args.gpu)
            if self.discriminator is not None:
                self.discriminator.to_gpu(args.gpu)

        # step 3 prepare data
        # determine whether rgb or black and white images have been used during training
        image_mode = log_data.get('image_mode', 'RGB')

        if args.eval_gt.endswith('.json'):
            args.eval_data = load_train_paths(args.eval_gt, with_label=True)
        else:
            args.eval_data = args.eval_gt

        self.data_loader = LabeledImageDataset(
            args.eval_data,
            root=os.path.dirname(args.eval_gt),
            image_size=self.image_size,
            image_mode=image_mode
        )
        if args.num_samples is not None:
            self.data_loader.shrink_dataset(args.num_samples)

        self.data_iterator = chainer.iterators.MultiprocessIterator(
            self.data_loader,
            args.batchsize,
            repeat=False,
            shuffle=False
        )

        # step 4 build bbox plotter in order to see eval result
        self.bbox_plotter = BBOXPlotter(
            self.data_loader.get_example(0)[0],
            os.path.join(args.model_dir, 'eval_bboxes'),
            self.target_size,
            render_extracted_rois=True,
            device=args.gpu,
            num_rois_to_render=4,
            show_visual_backprop_overlay=False,
            show_backprop_and_feature_vis=True,
            visualization_anchors=[
                ["visual_backprop_anchors"],
            ],
        )
        self.bbox_plotter.xp = self.localizer.xp

        # add some fields for accuracy calculation
        if self.args.deteval:
            self.deteval_xml_tree_root = ET.Element('tagset')

        self.num_hits = 0
        self.num_objects = 0
        self.num_predicted_objects = 0
        with cuda.Device(self.args.gpu):
            self.bad_ious = self.localizer.xp.array((0,), dtype='f')

        self.results_path = os.path.join(self.args.model_dir, 'eval_results.json')

    def load_weights(self, snapshot_name, model):
        with np.load(os.path.join(self.args.model_dir, snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f).load(model)

    def reset(self):
        self.num_hits = 0
        self.num_objects = 0
        self.num_predicted_objects = 0

        with cuda.Device(self.args.gpu):
            self.bad_ious = self.localizer.xp.array((0,), dtype='f')
        self.data_iterator.reset()

    def load_module(self, module_file):
        module_spec = importlib.util.spec_from_file_location("models.model", module_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def postprocess_with_nms(self, rois, bboxes, objectness_scores, image_size):
        xp = self.localizer.xp
        # bring bboxes into correct data format
        nms_bboxes = get_aabb_corners(bboxes, image_size)
        nms_bboxes = xp.stack([b.data for b in [nms_bboxes[1], nms_bboxes[0], nms_bboxes[3], nms_bboxes[2]]]).transpose(1, 0)
        # determine scores
        nms_objectness_scores = F.softmax(objectness_scores)
        # filter nms bboxes
        indices_to_keep = xp.nonzero(F.argmax(nms_objectness_scores, axis=1).data)[0]
        # indices_to_keep = xp.arange(len(nms_objectness_scores))
        nms_bboxes = nms_bboxes[indices_to_keep]
        nms_objectness_scores = nms_objectness_scores[indices_to_keep][:, 1].data
        indices = non_maximum_suppression(nms_bboxes, 0.2, score=nms_objectness_scores)
        indices = [int(indices_to_keep[int(i)]) for i in indices]
        return rois[indices], bboxes[indices], objectness_scores[indices]

    def add_image_to_deteval_xml(self, image_name, image_size, bboxes):
        image_node = ET.SubElement(self.deteval_xml_tree_root, 'image')
        image_name_element = ET.SubElement(image_node, 'imageName')
        image_name_element.text = f"{image_name}.png"
        rectangle_list = ET.SubElement(image_node, 'taggedRectangles')

        bboxes = get_aabb_corners(bboxes, image_size)
        bboxes = self.localizer.xp.stack([b.data for b in bboxes]).transpose(1, 0)

        x_all = bboxes[:, 0]
        y_all = bboxes[:, 1]
        width_all = bboxes[:, 2] - bboxes[:, 0]
        height_all = bboxes[:, 3] - bboxes[:, 1]

        for x, y, width, height in zip(x_all, y_all, width_all, height_all):
            ET.SubElement(rectangle_list, 'taggedRectangle', attrib={
                "x": str(x),
                "y": str(y),
                "width": str(width),
                "height": str(height),
            })

    def calc_accuracy(self, predicted_bboxes, gt_bboxes, image_size):
        xp = self.localizer.xp
        self.num_objects += len(gt_bboxes)
        self.num_predicted_objects += len(predicted_bboxes)

        if len(predicted_bboxes) == 0:
            return

        predicted_bboxes = get_aabb_corners(predicted_bboxes, image_size)
        predicted_bboxes = xp.stack(
            [b.data for b in predicted_bboxes]
        ).transpose(1, 0)

        all_ious = []
        for gt_bbox in gt_bboxes:
            gt_bbox = xp.tile(gt_bbox, (len(predicted_bboxes), 1))
            ious = bbox_iou(gt_bbox, predicted_bboxes)
            all_ious.append(ious)

            # a predicted bbox is correct, iff its iou with the groundtruth bbox is higher than the given threshold
            good_bboxes = xp.where((ious[0] >= self.args.iou_threshold))
            if len(good_bboxes[0]) == 0:
                self.bad_ious = xp.concatenate((self.bad_ious, ious[0, ious[0].nonzero()[0]]), axis=0)
                continue
            self.num_hits += 1
        return all_ious

    def evaluate(self, snapshot_name=''):
        current_device = cuda.get_device_from_id(self.args.gpu)
        predictions = []
        gt_data = []
        with current_device:
            for i, batch in enumerate(tqdm(self.data_iterator, total=len(self.data_loader) // self.args.batchsize)):
                image, gt_bboxes, gt_labels = batch[0]
                gt_data.append((gt_bboxes, gt_labels))
                image_size = Size._make(image.shape[-2:])
                if self.args.gpu is not None:
                    image = cuda.to_gpu(image, current_device)

                with cuda.Device(self.args.gpu):
                    with configuration.using_config('train', False):
                        rois, bboxes = self.localizer(image.copy()[None, ...])[:2]

                        if self.discriminator is not None:
                            class_predictions = self.discriminator(rois)
                        else:
                            class_predictions = None

                    if len(rois.shape) > 4:
                        rois = self.localizer.xp.reshape(rois.data, (-1,) + rois.shape[2:])
                    else:
                        rois = rois.data

                    if len(bboxes.shape) > 4:
                        bboxes = self.localizer.xp.reshape(bboxes.data, (-1,) + bboxes.shape[2:])
                    else:
                        bboxes = bboxes.data
                    predictions.append((cuda.to_cpu(F.stack(get_aabb_corners(bboxes, image_size), axis=1).data),))

                    backprop_visualizations = self.bbox_plotter.get_backprop_visualization(self.localizer)

                    ious = self.calc_accuracy(bboxes.copy(), gt_bboxes, image_size)

                    if self.args.save_predictions:
                        self.save_rois(gt_bboxes, backprop_visualizations, bboxes, class_predictions, i, image, rois, ious)

                    if self.args.deteval:
                        self.add_image_to_deteval_xml(i, image_size, bboxes.copy())

            if self.args.deteval:
                rough_xml_string = ET.tostring(self.deteval_xml_tree_root, encoding='utf-8')
                pretty_xml = minidom.parseString(rough_xml_string).toprettyxml(encoding='utf-8').decode('utf-8')
                with open(os.path.join(self.args.model_dir, 'deteval.xml'), 'w') as destination:
                    destination.write(pretty_xml)

            self.save_eval_results(snapshot_name, predictions, gt_data)

    def save_rois(self, gt_bboxes, backprop_visualizations, bboxes, class_predictions, index, image, rois, ious):
        dest_image = self.bbox_plotter.render_rois(
            rois,
            bboxes.copy(),
            index,
            image,
            backprop_vis=backprop_visualizations,
        )
        if class_predictions is not None:
            dest_image = self.bbox_plotter.render_discriminator_result(
                dest_image,
                self.bbox_plotter.array_to_image(image.copy()),
                self.bbox_plotter.get_discriminator_output_function(class_predictions)
            )
        if self.args.render_gt:
            draw = ImageDraw.Draw(dest_image)
            for i, (gt_bbox, iou) in enumerate(zip(gt_bboxes, ious), start=1):
                corners = [
                    (gt_bbox[1], gt_bbox[0]),  # top-left
                    (gt_bbox[3], gt_bbox[0]),  # top-right
                    (gt_bbox[3], gt_bbox[2]),  # bottom-right
                    (gt_bbox[1], gt_bbox[2]),  # bottom-left
                ]
                self.bbox_plotter.draw_bbox("red", corners, draw)
                iou = format(float(np.max(cuda.to_cpu(iou)[0])), '.3')
                dest_image = self.bbox_plotter.render_text(dest_image, self.bbox_plotter.array_to_image(image.copy()), iou, i)
        self.bbox_plotter.save_image(dest_image, index)

    def save_eval_results(self, snapshot_name, predictions, gt_data):
        if self.args.save_predictions:
            # we are not doing a real evaluation, we want to have a look at predictions
            return

        # calculate map for our detection
        predicted_bboxes = concat_examples(predictions)[0]
        pred_scores = np.ones((len(predicted_bboxes), 1))
        pred_labels = np.zeros_like(pred_scores)
        gt_bboxes, gt_labels = concat_examples(gt_data)

        result = chainercv.evaluations.eval_detection_voc(
            predicted_bboxes,
            pred_labels,
            pred_scores,
            gt_bboxes,
            gt_labels
        )

        recall = self.num_hits / self.num_objects
        precision = self.num_hits / self.num_predicted_objects
        if precision + recall != 0:
            h_mean = 2 * (precision * recall) / (precision + recall)
        else:
            h_mean = 0.0

        if os.path.exists(self.results_path):
            with open(self.results_path) as eval_file:
                json_data = json.load(eval_file)
        else:
            json_data = []

        json_data.append({
            "ap": result["map"],
            "recall": recall,
            "precision": precision,
            "h_mean": h_mean,
            "bad_iou_mean": float(self.bad_ious.mean()),
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
    best_epoch = np.argmax(np.array(values_per_key['ap']))
    print(f"best ap: {max(values_per_key['ap'])}")
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
    parser.add_argument("--use-nms", action='store_true', default=False, help="post process prediction with NMS")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold indicating if a predicted bbox is correct, based on its iou with gt [default: 0.7]")
    parser.add_argument("--save-predictions", action='store_true', default=False, help="use bbox plotter to store the predicted bboxes for every test sample")
    parser.add_argument("--deteval", action='store_true', default=False, help="produce an xml file that can be used together with the deteval tool")
    parser.add_argument("--assessor", help="name of discriminator to use")
    parser.add_argument("--render-gt", action='store_true', default=False, help="render gt bbox into resulting image (should be used in conjunction with `save-predictions`")
    parser.add_argument("--force-reset", action='store_true', default=False, help="force a reset of eval results file")

    args = parser.parse_args()

    evaluator = Evaluator(args)
    if os.path.exists(evaluator.results_path) and not args.save_predictions:
        if args.force_reset:
            os.unlink(evaluator.results_path)
            evaluated_snapshots = []
        else:
            # we already evaluated some snapshots, so we do not need to do that again
            with open(evaluator.results_path) as already_evaluated_model_results:
                json_data = json.load(already_evaluated_model_results)
                evaluated_snapshots = [item['snapshot_name'] for item in json_data]
    else:
        evaluated_snapshots = []

    snapshots = list(sorted(filter(lambda x: x not in evaluated_snapshots and args.snapshot_prefix in x, os.listdir(args.model_dir)), key=lambda x: int(getattr(re.search(r"(\d+).npz", x), 'group', lambda: 0)(1))))
    for snapshot in tqdm(snapshots):
        try:
            evaluator.load_weights(snapshot, evaluator.localizer)
            evaluator.reset()
            evaluator.evaluate(snapshot)
        except Exception as e:
            print(f"Exception: {e} at snapshot: {snapshot}")

    if os.path.exists(evaluator.results_path):
        with open(evaluator.results_path) as evaluated_model_results:
            json_data = json.load(evaluated_model_results)
            plot_eval_results(json_data, args.model_dir)
