import glob

import numpy
import scipy.stats
from tqdm import tqdm

import pandas
import math
import argparse
import numpy as np

from evaluation_metrics.poses_evaluation import get_2d_landmark_diffs


def get_non_mean_distance_of_img(keypoints_a, keypoints_b, normalization):
    """ take two arrays, containing all the keypoints for all the images in a dataset. For each image, check if the
    keypoints are not None, and return the raw distance diffs for those cases.

    recompute the distances between all the keypoints of each image and return them"""

    if normalization in ["left_hand", "right_hand"]:
        normalization = "hands"

    all_diffs = []

    # for each image, check if the keypoints were found or were None
    for img_keypoints_a, img_keypoints_b in zip(keypoints_a, keypoints_b):
        if img_keypoints_a is not None and img_keypoints_b is not None:
            current_img_diffs = get_2d_landmark_diffs(img_keypoints_a, img_keypoints_b, normalization)
        elif img_keypoints_a is None and img_keypoints_b is None:
            current_img_diffs = math.nan
        elif img_keypoints_a is None and img_keypoints_b is not None:
            # if the model identifies this feature in the modified image but not in the original img
            current_img_diffs = math.inf
        elif img_keypoints_a is not None and img_keypoints_b is None:
            # if the model identifies this feature in the original image but not in the modified img
            current_img_diffs = -math.inf
        else:
            raise "the if else logic of the landmarks is wrong, we should never reach this line"
        all_diffs.append(current_img_diffs)
    all_diffs = pandas.Series(all_diffs)

    return all_diffs


def get_different_per_img_measurements_dict(all_raw_distances, per_image_measurements):
    """Given an array with all the distances between two series of keypoints for all images in a dataset, return
    an array with a per_image_aggregation of the distances different than the mean, for example the max, or the mean
    of the top 5"""

    all_max = []
    all_top_5 = []
    for per_img_distances in all_raw_distances:

        if type(per_img_distances) != numpy.ndarray:
            # the special values (math.nan, math.inf) we leave as they are
            all_max.append(per_img_distances)
            all_top_5.append(per_img_distances)
        else:
            # we actually compute the max, top5, etc.

            # overall biggest keypoint error for each image
            all_max.append(np.max(per_img_distances))

            # mean of top 5 outliers
            max_5_indxs = np.argpartition(per_img_distances, -5)[-5:]
            max_5_vals = per_img_distances[max_5_indxs]
            max_5_means = np.sum(max_5_vals) / 5
            all_top_5.append(max_5_means)

    per_image_measurements["max"] = pandas.Series(all_max)
    per_image_measurements["top_5_outliers"] = pandas.Series(all_top_5)

    return per_image_measurements


def aggregate_results(input_filepath, output_path):
    aggregations = pandas.DataFrame(columns=["run_name", "met_rule", "results_type", "per_img_measurement", "mean",
                                             "std", "min", "25%", "50%", "75%", "max", "nans", "inf", "-inf",
                                             "both_detected_count", "total_imgs"]+[f"cdf{num/10}" for num in range(1, 10)])

    # aggregate all the metRules configurations for the input dataset and data_type
    input_filepaths = sorted(list(glob.glob(f"{input_filepath}/*")))

    # grab the no-rule system_outs, for when we need to recompute the diffs with other rules
    no_rule_file = [filepath for filepath in input_filepaths if "no-rule-orig-landmarks" in filepath][0]
    no_rule_table = pandas.read_pickle(no_rule_file)
    no_rule_table = no_rule_table.set_index(pandas.Index(range(len(no_rule_table))))
    # split the diffs dict into different columns
    no_rule_outputs = pandas.json_normalize(no_rule_table['system_outs'])

    for file_path in tqdm(input_filepaths, desc=f"aggregating results for {input_filepath}"):
        table = pandas.read_pickle(file_path)
        table = table.set_index(pandas.Index(range(len(table))))

        # split the diffs dict into different columns
        split_diffs = table.join(pandas.json_normalize(table['eval_diffs']))

        # keep only the rows that show the diffs and keypoints of modified images, not the original ones
        rows_with_diff = split_diffs[split_diffs.met_rules != "no-rule-orig-landmarks"]

        if 'no-rule-orig-landmarks' in file_path:
            diffs_keys = []  # # for the no metRule, we only care about diff_with_ground_truth
        else:
            # get the names of the eval_diffs (hands, face, pose, etc.)
            diffs_keys = rows_with_diff['eval_diffs'].apply(lambda x: list(x.keys())).iloc[0]
        # for the datasets that have ground truth, add it to the keys of diffs to analyze
        if 'FLIC' in file_path:
            diffs_keys.append("diff_with_gt")

        this_rule_outputs = pandas.json_normalize(table['system_outs'])
        for met_rule, same_met_rule_group in split_diffs[["met_rules"]+diffs_keys].groupby(['met_rules']):
            met_rule = met_rule[0]

            # iterate over the columns of this dataframe, which are the different kinds of diffs we have.
            for diff_type, current_diffs in same_met_rule_group.items():
                if diff_type == "met_rules":
                    continue  # we are doing one row per diff metric, we don't need a row just to write the met_rule
                if met_rule == 'no-rule-orig-landmarks' and diff_type != 'diff_with_gt':
                    continue  # for the no metRule, we only care about diff_with_ground_truth
                if 'phoenix' in file_path and diff_type == 'diff_with_gt':
                    continue  # Some datasets don't have ground truth, like phoenix

                # get more values than just the mean across images, more can be added
                per_image_measurements = {"mean": current_diffs}  # the default value saved in the dataframe is the mean for each image
                if diff_type in ["diff_with_gt", "pose"]: # the metrics computed with pck3d can only count percentage of correct joints
                    pass
                else:
                    assert diff_type in ["left_hand", "right_hand", "face"]
                    raw_distance_diffs_per_image = get_non_mean_distance_of_img(no_rule_outputs[diff_type],
                                                                                this_rule_outputs[diff_type],
                                                                                diff_type, )
                    per_image_measurements = get_different_per_img_measurements_dict(raw_distance_diffs_per_image,
                                                                                     per_image_measurements)

                for per_image_measurement, current_measurement_diffs in per_image_measurements.items():
                    # we have different possible measurements of how to aggregate all the diff values for each image.
                    # Of all keypoints in an image, we aggregate to one number, the mean, max, etc. of all the diffs
                    # and then aggregate some values about this per image value accross all images.

                    # count nans, infs and -infs as separate aggregation and remove them before computing means and percentiles
                    values_count = current_measurement_diffs.value_counts(dropna=False)
                    special_vals = {math.nan: 0, math.inf: 0, -math.inf: 0}
                    for val in [math.nan, math.inf, -math.inf]:
                        if val in values_count:
                            special_vals[val] = values_count[val]

                    # after counting special_vals +-infs, replace them with nans to facilitate other metrics counting
                    current_measurement_diffs.replace([math.inf, -math.inf], math.nan, inplace=True)

                    descriptions = current_measurement_diffs.describe()
                    descriptions.loc["nans"] = special_vals[math.nan]
                    descriptions.loc["inf"] = special_vals[math.inf]
                    descriptions.loc["-inf"] = special_vals[-math.inf]
                    descriptions.loc["total_imgs"] = (special_vals[math.nan] + special_vals[math.inf] +
                                                      special_vals[-math.inf] + descriptions["count"])
                    descriptions.rename({"count": "both_detected_count"}, inplace=True)

                    new_row = descriptions.to_dict()

                    # count what percentage of images have a diff of at least X, where X = min_diff + [0.1..1] * (max_diff - min_diff)
                    # i.e. the cumulative distribution. We don't have problems with +-math.inf because they were removed before
                    diff_thresholds = [new_row["min"] + (thresh / 10) * (new_row["max"] - new_row["min"]) for thresh in
                                       range(1, 10)]
                    cumulative_distribution_values = scipy.stats.percentileofscore(current_measurement_diffs, diff_thresholds,
                                                                                   nan_policy="omit")
                    for cumulative_distribution_idx in range(0, 9):
                        new_row.update({f"cdf{cumulative_distribution_idx/10}":
                                            cumulative_distribution_values[cumulative_distribution_idx]})

                    new_row.update({"met_rule": met_rule, "results_type": diff_type,
                                    "per_img_measurement": per_image_measurement})  # "run_name": table_file_name,
                    aggregations.loc[len(aggregations.index)] = new_row

    # save all the aggregations in a new csv together for later viewing
    aggregations.to_csv(output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--in_dir', nargs='+', type=str, required=True,
                        help='Path to folder with all the result files for different metRules '
                             'with the results to aggregate. The script will analyze all files in the --in_dir folder')
    parser.add_argument('-out', '--out_dir', type=str, required=False,
                        help='Path to folder where to save the csv. The name of the file will be'
                             '{--out_dir}+{in_dir}. If no --out_dir is provided, the output file will be {--in_dir}.csv')
    args = parser.parse_args()

    for input_dir in args.in_dir:
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = input_dir + "-aggregation.csv"  # save the results in the parent of input folder

        aggregate_results(input_dir, out_dir)



