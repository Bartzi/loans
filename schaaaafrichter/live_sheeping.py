import argparse
import queue

import cv2

from sheeping.asynchronous_sheep_localizer import AsynchronousSheepLocalizer
from sheeping.audio_renderer import Baaaer
from sheeping.camera import Camera

FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX


def print_fps(image, fps):
    image_height, image_width, _ = image.shape
    fps_text = "{fps:.2f} FPS".format(fps=fps)
    text_size = cv2.getTextSize(fps_text, FPS_FONT, 0.8, 1)[0]
    text_start = image_width - text_size[0], text_size[1]
    cv2.putText(image, fps_text, text_start, FPS_FONT, 0.8, (0, 255, 0), bottomLeftOrigin=False)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the HPI Sheep")
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("log_file", help="path to log file that has been used to train model")
    parser.add_argument("-c", "--camera", type=int, default=0, help="camera to use")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="id of gpu to use")

    args = parser.parse_args()

    camera = Camera(camera_id=args.camera)
    localizer = AsynchronousSheepLocalizer(args.model_file, args.log_file, args.gpu)
    localizer.start_localization_worker()

    baaaer = Baaaer()

    bboxes = scores = fps = None
    try:
        with camera:
            while True:
                frame = camera.get_frame()
                frame = cv2.flip(frame, 1)
                processed_frame, scaling = localizer.resize(frame)
                processed_frame = localizer.preprocess(processed_frame)
                try:
                    localizer.localization_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass

                try:
                    bboxes, scores, fps = localizer.image_queue.get_nowait()
                    if len(bboxes) > 0:
                        baaaer.baaa()
                except queue.Empty:
                    pass

                if bboxes is not None:
                    frame = localizer.visualize_results(frame, bboxes, scores, scaling)
                    frame = print_fps(frame, fps)

                cv2.imshow('sheeper', frame)
                pressed_key = cv2.waitKey(1) & 0xff
                if pressed_key == 27:
                    break  # quit with ESC
                elif pressed_key == 171:  # increase with +
                    localizer.score_threshold += 0.05
                    print("setting score threshold to: {:.2}".format(localizer.score_threshold))
                elif pressed_key == 173:  # decrease with -
                    localizer.score_threshold -= 0.05
                    print("setting score threshold to: {:.2}".format(localizer.score_threshold))
                elif pressed_key == ord('b'):  # toggle baa sound with b
                    baaaer.enabled = not baaaer.enabled
    finally:
        baaaer.shutdown()
        localizer.shutdown()
        cv2.destroyAllWindows()
