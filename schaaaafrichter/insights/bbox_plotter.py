import base64
import json

import os
import socket

from collections import namedtuple
from io import BytesIO

import numpy as np
from PIL import Image, ImageFont

from PIL import ImageDraw

import chainer
from chainer.backends import cuda
from chainer.training import Extension


Size = namedtuple("Size", ['height', 'width'])


COLOR_MAP = [
    "#00B3FF",  # Vivid Yellow
    "#753E80",  # Strong Purple
    "#0068FF",  # Vivid Orange
    "#D7BDA6",  # Very Light Blue
    "#2000C1",  # Vivid Red
    "#62A2CE",  # Grayish Yellow
    "#667081",  # Medium Gray

    # The following don't work well for people with defective color vision
    "#347D00",  # Vivid Green
    "#8E76F6",  # Strong Purplish Pink
    "#8A5300",  # Strong Blue
    "#5C7AFF",  # Strong Yellowish Pink
    "#7A3753",  # Strong Violet
    "#008EFF",  # Vivid Orange Yellow
    "#5128B3",  # Strong Purplish Red
    "#00C8F4",  # Vivid Greenish Yellow
    "#0D187F",  # Strong Reddish Brown
    "#00AA93",  # Vivid Yellowish Green
    "#153359",  # Deep Yellowish Brown
    "#133AF1",  # Vivid Reddish Orange
    "#162C23",  # Dark Olive Green
]


def get_next_color():
    while True:
        for color in COLOR_MAP:
            yield color


class BBOXPlotter(Extension):

    def __init__(self, image, out_dir, **kwargs):
        super(BBOXPlotter, self).__init__()
        self.image = image.copy()
        self.image_size = Size(height=image.shape[1], width=image.shape[2])
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.colours = get_next_color
        self.send_bboxes = kwargs.pop("send_bboxes", False)
        self.upstream_ip = kwargs.pop("upstream_ip", '127.0.0.1')
        self.upstream_port = kwargs.pop("upstream_port", 1337)
        self.font = ImageFont.truetype("insights/DejaVuSans.ttf", 20)
        self.xp = np
        self.devices = kwargs.pop('devices', None)

    def initialize(self, trainer):
        # run the network with the completely randomized state we start with
        self(trainer)

    def send_image(self, data):
        height = data.height
        width = data.width
        channels = len(data.getbands())

        # convert image to png in order to save network bandwidth
        png_stream = BytesIO()
        data.save(png_stream, format="PNG")
        png_stream = png_stream.getvalue()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect((self.upstream_ip, self.upstream_port))
            except Exception as e:
                print(e)
                print("could not connect to display server, disabling image rendering")
                self.send_bboxes = False
                return
            data = {
                'width': width,
                'height': height,
                'channels': channels,
                'image': base64.b64encode(png_stream).decode('utf-8'),
            }
            sock.send(bytes(json.dumps(data), 'utf-8'))

    def array_to_image(self, array):
        if array.shape[0] == 1:
            # image is black and white, we need to trick the system into thinking, that we are having an RGB image
            array = self.xp.tile(array, (3, 1, 1))
        return Image.fromarray(cuda.to_cpu(array.astype(np.uint8).transpose(1, 2, 0))).convert("RGBA")

    def variable_to_image(self, data):
        if isinstance(data, chainer.Variable):
            data = data.data
        return self.array_to_image(data)

    def __call__(self, trainer):
        iteration = trainer.updater.iteration

        with cuda.get_device_from_id(trainer.updater.get_optimizer('main').target._device_id), chainer.using_config('train', False):
            self.xp = np
            image = self.xp.asarray(self.image)
            predictor = trainer.updater.get_optimizer('main').target
            predictor.model.use_preset('visualize')
            bboxes, labels, scores = predictor.model.predict([image])
            predictor.model.use_preset('evaluate')

            image = self.array_to_image(self.image.copy())
            self.render_image_and_bboxes(image, bboxes[0], scores[0])
            image.save("{}.png".format(os.path.join(self.out_dir, str(iteration))), 'png')
            if self.send_bboxes:
                self.send_image(image)

    def render_image_and_bboxes(self, base_image, bboxes, scores):
        if len(bboxes) == 0:
            return

        draw = ImageDraw.Draw(base_image)
        for bbox, score, colour in zip(bboxes, scores, self.colours()):
            bbox = np.clip(bbox, 0, self.image_size.width)

            top_left = bbox[1], bbox[0]
            top_right = bbox[1], bbox[2]
            bottom_left = bbox[3], bbox[0]
            bottom_right = bbox[3], bbox[2]

            corners = [top_left, top_right, bottom_right, bottom_left]
            next_corners = corners[1:] + [corners[0]]

            for first_corner, next_corner in zip(corners, next_corners):
                draw.line([first_corner, next_corner], fill=colour, width=3)
