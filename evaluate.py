import argparse
import os
import numpy as np
import tensorflow as tf
from glob import glob
import re
import csv
from collections import OrderedDict
import os
from Common import pc_util
from Common.pc_util import load, save_ply_property,get_pairwise_distance
from Common.ops import normalize_point_cloud
from tf_ops.nn_distance import tf_nndistance
from sklearn.neighbors import NearestNeighbors
import math
from time import time
parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, default="./result/Ours/", help="pred points")
parser.add_argument("--gt", type=str, default="./data/test/gt_sample_8192/", help="gt points")
FLAGS = parser.parse_args()

PRED_DIR = os.path.abspath(FLAGS.pred)
GT_DIR = os.path.abspath(FLAGS.gt)

gt_paths = glob(os.path.join(GT_DIR,'*.xyz'))
gt_names = [os.path.basename(p)[:-4] for p in gt_paths]

print(PRED_DIR)
print(GT_DIR)
print(len(gt_paths))
print(gt_names)


gt = load(gt_paths[0])[:, :3]
pred_placeholder = tf.placeholder(tf.float32, [1, gt.shape[0], 3])
gt_placeholder = tf.placeholder(tf.float32, [1, gt.shape[0], 3])
pred_tensor, centroid, furthest_distance = normalize_point_cloud(pred_placeholder)
gt_tensor, centroid, furthest_distance = normalize_point_cloud(gt_placeholder)

cd_forward, _, cd_backward, _ = tf_nndistance.nn_distance(pred_tensor, gt_tensor)
cd_forward = cd_forward[0, :]
cd_backward = cd_backward[0, :]

precentages = np.array([0.004, 0.006,0.008, 0.010, 0.012])

with tf.Session() as sess:
    fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"]

    print("{:60s} ".format("name"), "|".join(["{:>15s}".format(d) for d in fieldnames[1:]]))
    for D in [PRED_DIR]:
        avg_md_forward_value = 0
        avg_md_backward_value = 0
        avg_hd_value = 0
        avg_emd_value = 0
        counter = 0
        pred_paths = glob(os.path.join(D, "*_2048.xyz"))

        gt_pred_pairs = []
        for p in pred_paths:
            name, ext = os.path.splitext(os.path.basename(p))
            name = name.replace('2048','8192')

            assert(ext in (".ply", ".xyz"))
            try:
                gt = gt_paths[gt_names.index(name)]
            except ValueError:
                pass
            else:
                gt_pred_pairs.append((gt, p))

        print("total inputs ", len(gt_pred_pairs))
        tag = re.search("/(\w+)/result", os.path.dirname(gt_pred_pairs[0][1]))
        if tag:
            tag = tag.groups()[0]
        else:
            tag = D

        print("{:60s}".format(tag), end=' ')
        global_p2f = []
        global_density = []

        with open(os.path.join(os.path.dirname(gt_pred_pairs[0][1]), "evaluation.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
            writer.writeheader()
            for gt_path, pred_path in gt_pred_pairs:
                row = {}
                gt = load(gt_path)[:, :3]
                gt = gt[np.newaxis, ...]
                pred = pc_util.load(pred_path)
                pred = pred[:, :3]

                row["name"] = os.path.basename(pred_path)
                pred = pred[np.newaxis, ...]
                cd_forward_value, cd_backward_value = sess.run([cd_forward, cd_backward], feed_dict={pred_placeholder:pred, gt_placeholder:gt})

                md_value = np.mean(cd_forward_value)+np.mean(cd_backward_value)
                hd_value = np.max(np.amax(cd_forward_value, axis=0)+np.amax(cd_backward_value, axis=0))
                cd_backward_value = np.mean(cd_backward_value)
                cd_forward_value = np.mean(cd_forward_value)
                row["CD"] = cd_forward_value+cd_backward_value
                row["hausdorff"] = hd_value
                avg_md_forward_value += cd_forward_value
                avg_md_backward_value += cd_backward_value
                avg_hd_value += hd_value
                if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
                    point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
                    if point2mesh_distance.size == 0:
                        continue
                    point2mesh_distance = point2mesh_distance[:, 3]
                    row["p2f avg"] = np.nanmean(point2mesh_distance)
                    row["p2f std"] = np.nanstd(point2mesh_distance)
                    global_p2f.append(point2mesh_distance)


                writer.writerow(row)
                counter += 1


            row = OrderedDict()

            avg_md_forward_value /= counter
            avg_md_backward_value /= counter
            avg_hd_value /= counter
            avg_emd_value /= counter
            avg_cd_value = avg_md_forward_value + avg_md_backward_value
            row["CD"] = avg_cd_value
            row["hausdorff"] = avg_hd_value
            row["EMD"] = avg_emd_value
            if global_p2f:
                global_p2f = np.concatenate(global_p2f, axis=0)
                mean_p2f = np.nanmean(global_p2f)
                std_p2f = np.nanstd(global_p2f)
                row["p2f avg"] = mean_p2f
                row["p2f std"] = std_p2f

            writer.writerow(row)
            print("|".join(["{:>15.8f}".format(d) for d in row.values()]))

