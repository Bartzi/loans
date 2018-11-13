import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def setup_video_writer(video_reader, output_name):
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    codec = int(video_reader.get(cv2.CAP_PROP_FOURCC))
    video_writer = cv2.VideoWriter(
        output_name,
        codec,
        video_fps,
        (video_width, video_height),
    )
    return video_writer


def bgr_to_rgb(frame):
    b, g, r = np.split(frame, 3, axis=2)
    return np.concatenate((r, g, b), axis=2)


def rgb_to_bgr(frame):
    r, g, b = np.split(frame, 3, axis=2)
    return np.concatenate((b, g, r), axis=2)


def blend_frames(base_frame, blend_frame, blend_alpha):
    base_frame = bgr_to_rgb(base_frame)
    blend_frame = bgr_to_rgb(blend_frame)

    base_image = Image.fromarray(base_frame)
    blend_image = Image.fromarray(blend_frame)

    blended_image = Image.blend(base_image, blend_image, blend_alpha)
    return rgb_to_bgr(np.asarray(blended_image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="take two videos and blend them together with a given blen factor")
    parser.add_argument("base_video", help="the base video")
    parser.add_argument("blend_video", help="the video that shall be blended on top of base video")
    parser.add_argument("output", help="path to output video")
    parser.add_argument("--blend-alpha", type=float, default=0.7, help="alpha value for blending")

    args = parser.parse_args()

    base_video = cv2.VideoCapture(args.base_video)
    blend_video = cv2.VideoCapture(args.blend_video)

    out_video = setup_video_writer(base_video, args.output)

    error_message = "Framecount of both videos is not equal!"
    assert base_video.get(cv2.CAP_PROP_FRAME_COUNT) == blend_video.get(cv2.CAP_PROP_FRAME_COUNT), error_message

    number_of_frames = int(base_video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(number_of_frames)):
        if not base_video.isOpened() or not blend_video.isOpened():
            break

        ret, base_frame = base_video.read()
        if ret is False:
            break

        ret, blend_frame = blend_video.read()
        if ret is False:
            break

        out_frame = blend_frames(base_frame, blend_frame, args.blend_alpha)

        out_video.write(out_frame)
