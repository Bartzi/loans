import argparse

import cv2
import os
from tqdm import tqdm

from sheep.unsupervised_sheep_localizer import UnsupervisedSheepLocalizer


def setup_video_reader_and_writer(args, video_name):
    video_reader = cv2.VideoCapture(video_name)
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    if args.visual_backprop:
        # use a different output to not overwrite old video_analysis
        video_name, extension = os.path.splitext(os.path.basename(video_name))
        video_name = f"{video_name}_visual_backprop{extension}"
        output_name = os.path.join(args.output, video_name)
    else:
        output_name = os.path.join(args.output, os.path.basename(video_name))

    codec = int(video_reader.get(cv2.CAP_PROP_FOURCC))
    video_writer = cv2.VideoWriter(
        output_name,
        codec,
        video_fps,
        (video_width, video_height),
    )
    return video_reader, video_writer


def sheep(args, localizer, video_name):
    video_reader, video_writer = setup_video_reader_and_writer(args, video_name)
    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for _ in tqdm(range(number_of_frames)):
            if not video_reader.isOpened():
                break

            ret, frame = video_reader.read()
            if ret is False:
                break

            resized_image, scaling = localizer.resize(frame)
            processed_image = localizer.preprocess(resized_image, bgr_to_rgb=True)

            bboxes, scores, visual_backprop = localizer.localize(processed_image,
                                                                 return_visual_backprop=args.visual_backprop)

            if visual_backprop is not None:
                visual_backprop = cv2.resize(visual_backprop, (frame.shape[:2:][::-1]), cv2.INTER_LANCZOS4)
                out_image = localizer.visualize_results(
                    visual_backprop,
                    bboxes,
                    scores,
                    scaling=scaling,
                    render_scores=args.assessor is not None
                )
            else:
                out_image = localizer.visualize_results(
                    frame,
                    bboxes,
                    scores,
                    scaling=scaling,
                    render_scores=args.assessor is not None
                )

            if visual_backprop is not None or args.visual_backprop is False:
                video_writer.write(out_image)
    finally:
        video_reader.release()
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the HPI Sheep in a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("log_file", help="path to log file that has been used to train model")
    parser.add_argument("-i", "--input_videos", nargs='+', required=True, help="path to recorded videos that shall be analyzed")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="id of gpu to use")
    parser.add_argument("-t", "--score-threshold", type=float, default=0.3, help="when to recognize a sheep")
    parser.add_argument("--assessor", help="path to trained discriminator that is used to predict confidence scores")
    parser.add_argument("-v", "--visual-backprop", action='store_true', default=False, help="render visual backprop view instead of normal view")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/predictions/videos",
        help="where images with predictions should be saved"
    )

    args = parser.parse_args()

    localizer_class = UnsupervisedSheepLocalizer
    localizer = localizer_class(args.model_file, args.log_file, args.gpu, discriminator=args.assessor)
    localizer.score_threshold = args.score_threshold

    os.makedirs(args.output, exist_ok=True)

    for video in tqdm(args.input_videos):
        sheep(args, localizer, video)
