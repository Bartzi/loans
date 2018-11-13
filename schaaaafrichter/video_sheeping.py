import argparse
import cv2
import os

from sheeping.sheep_localizer import SheepLocalizer


def setup_video_reader_and_writer(args):
    video_reader = cv2.VideoCapture(args.input_video)
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    codec = int(video_reader.get(cv2.CAP_PROP_FOURCC))
    video_writer = cv2.VideoWriter(
        os.path.join(args.output, os.path.basename(args.input_video)),
        codec,
        video_fps,
        (video_width, video_height),
    )
    return video_reader, video_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the HPI Sheep in a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("log_file", help="path to log file that has been used to train model")
    parser.add_argument("input_video", help="path to recorded video that shall be analyzed")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="id of gpu to use")
    parser.add_argument("-t", "--score-threshold", type=float, default=0.3, help="when to recognize a sheep")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/predictions/videos",
        help="where images with predictions should be saved"
    )

    args = parser.parse_args()

    localizer = SheepLocalizer(args.model_file, args.log_file, args.gpu)
    localizer.score_threshold = args.score_threshold

    os.makedirs(args.output, exist_ok=True)

    video_reader, video_writer = setup_video_reader_and_writer(args)
    number_of_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    try:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if ret is False:
                break

            resized_image, scaling = localizer.resize(frame)
            processed_image = localizer.preprocess(resized_image)
            bboxes, scores = localizer.localize(processed_image)

            out_image = localizer.visualize_results(frame, bboxes, scores, scaling=scaling)
            video_writer.write(out_image)
            print("processed frame {:5}/{}".format(frame_num, number_of_frames), end='\r')
            frame_num += 1
    finally:
        video_reader.release()
        video_writer.release()
        cv2.destroyAllWindows()
