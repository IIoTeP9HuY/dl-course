#!/usr/bin/env python

import argparse
import numpy as np

def convert(input_dataset, output_dataset):
    samples = []
    samples_processed = 0
    with open(input_dataset, 'r') as f:
        while True:
            try:
                samples.append(np.load(f))
                samples_processed += 1
            except Exception, e:
                break
            if samples_processed % 10000 == 0:
                print("Processed: {}".format(samples_processed))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset", help="path to input dataset")
    parser.add_argument("output_dataset", help="path to output dataset")

    args = parser.parse_args()

    convert(args.input_dataset, args.output_dataset)

if __name__ == "__main__":
    main()
