import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take a csv with iou gt and create a histogram out of gt labels")
    parser.add_argument('csv_file')
    parser.add_argument('-b', '--bins', default=10, type=int, help='number of bins for histogram')

    args = parser.parse_args()

    with open(args.csv_file) as handle:
        reader = csv.reader(handle, delimiter='\t')
        data = [float(l[1]) for l in reader]

    n, bins, patches = plt.hist(data, args.bins)
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(args.csv_file), 'histogram.png'))
