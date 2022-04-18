# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import pickle

import numpy as np
import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir", required=True, type=str, help="Path to extracted features dir"
    )
    parser.add_argument(
        "--out-file", required=True, type=str
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    infiles = glob.glob(os.path.join(args.features_dir, "*.npy"))

    data = {}
    for infile in tqdm.tqdm(infiles):
        reader = np.load(infile, allow_pickle=True)
        item = {}
        item["image_id"] = reader.item().get("image_id") + ".jpg"
        img_id = str(item["image_id"])
        item["img_h"] = reader.item().get("image_height")
        item["img_w"] = reader.item().get("image_width")
        item["num_boxes"] = reader.item().get("num_boxes")
        item["boxes"] = reader.item().get("bbox")
        item["features"] = reader.item().get("features")

        data[img_id] = item

    pickle.dump(data, open(args.out_file, "wb"))
