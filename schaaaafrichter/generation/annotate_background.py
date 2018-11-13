import argparse

from viewer import Viewer


def main(args):
    viewer = Viewer(args.output_folder)
    viewer.images = args.image
    viewer.master.title('Insert bounding boxes')
    viewer.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create simple bounding box annotations for a simple neural network")
    parser.add_argument("image", nargs="+", help="image files to annotate")
    parser.add_argument("--output-folder", default="data/bounding_boxes", help="folder where bounding box files should be saved")

    main(parser.parse_args())
