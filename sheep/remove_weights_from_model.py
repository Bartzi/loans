import argparse
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove keys from npz file")
    parser.add_argument("npz_file", help="path to npz file that shall be adjusted")
    parser.add_argument("-k", "--key_prefix", help="prefix of keys to remove", default="param_predictor")

    args = parser.parse_args()

    print("loading model")
    with np.load(args.npz_file) as handle:
        data = dict(handle)

        keys_to_delete = list(filter(lambda x: args.key_prefix in x, handle.keys()))
    print(f"removing the following keys: {keys_to_delete}")

    for key in keys_to_delete:
        del data[key]

    destination_path = f"{os.path.splitext(args.npz_file)[0]}.{args.key_prefix}-stripped"
    print(f"saving model to {destination_path}")
    np.savez(destination_path, **data)
