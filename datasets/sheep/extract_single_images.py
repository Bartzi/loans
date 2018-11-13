import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="take sheep gt as json an romve all images with more than one bbox")
    parser.add_argument("gt", help="path to gt file")
    parser.add_argument("output", help="path to new gt")

    args = parser.parse_args()

    with open(args.gt) as handle:
        gt_data = json.load(handle)

    bboxes_to_keep = list(filter(lambda x: len(x['bounding_boxes']) == 1, gt_data))

    with open(args.output, "w") as handle:
        json.dump(bboxes_to_keep, handle, indent=4)
