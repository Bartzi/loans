import chainer.functions as F
import chainercv

from chainer import reporter
from chainer.backends import cuda
from chainercv.utils import bbox_iou

from common.utils import Size


class SheepMAPEvaluator:

    def __init__(self, link, device):
        self.link = link
        self.device = device

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

    def __call__(self, *inputs):
        images, labels = inputs[:2]
        with cuda.Device(self.device):
            _, bboxes = self.link(images)

            bboxes = cuda.to_cpu(bboxes.data)
            labels = cuda.to_cpu(labels)

            xp = cuda.get_array_module(bboxes)

            bboxes = self.extract_corners(bboxes)
            bboxes = self.scale_bboxes(bboxes, Size._make(images.shape[-2:]))

            ious = bbox_iou(bboxes.data.copy(), xp.squeeze(labels))[xp.eye(len(bboxes)).astype(xp.bool)]
            mean_iou = ious.mean()

            reporter.report({'mean_iou': mean_iou})

            pred_bboxes = [bbox.data[xp.newaxis, ...].astype(xp.int32) for bbox in F.separate(bboxes, axis=0)]
            pred_scores = xp.ones((len(bboxes), 1))
            pred_labels = xp.zeros_like(pred_scores)

            gt_bboxes = [bbox.data[...] for bbox in F.separate(labels, axis=0)]
            gt_labels = xp.zeros_like(pred_scores)

            result = chainercv.evaluations.eval_detection_voc(
                pred_bboxes,
                pred_labels,
                pred_scores,
                gt_bboxes,
                gt_labels
            )

            reporter.report({'map': result['map']})
            reporter.report({'ap/sheep': result['ap'][0]})
