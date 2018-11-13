import argparse
import numpy as np


def main(model_in, model_out):
    with np.load(model_in) as data:
        prefix = "updater/model:main/model/"

        model_keys = filter(lambda x: prefix in x, data.keys())
        model_data = {key: data[key] for key in model_keys}
        renamed_data = {k.replace(prefix, ""): v for k, v in model_data.items()}

        np.savez(model_out, **renamed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert training snapshot to model file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_file", help="path to saved model")
    parser.add_argument("output_file", help="path to converted model")

    args = parser.parse_args()
    main(args.model_file, args.output_file)
