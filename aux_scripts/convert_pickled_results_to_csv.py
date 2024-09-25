import glob
import itertools
import json
import math
import os

import numpy
import pandas
from tqdm import tqdm

from evaluate_results import get_non_mean_distance_of_img
from test_systems.test_mediapipe_holistic import load_flic_images, diff_holistic_pose_and_flic_ground_truth


def mult_cols(row, cols):
    return row.tolist() if type(row) is numpy.ndarray else [row] * len(cols)


def get_video_and_image_name(img_string, dataset):
    # returns a tuple: (base_images_folder, video_name, image_name), if all are concatenated, the result is the full image path
    if dataset == "FLIC" or dataset == "FLIC-full":
        images_folder = img_string.rpartition("/")[0]
        video_and_image = "/" + img_string.rpartition("/")[2]
        video, _,  image = video_and_image.rpartition("-")
        image = "-" + image
    elif dataset == "phoenix":
        folder_and_video, _, image = img_string.rpartition("/")
        image = "/" + image
        images_folder, _,  video = folder_and_video.rpartition("/")
        video = "/" + video
    else:
        raise ValueError(f"Please specify how to process filenames from dataset {dataset}")
    return images_folder, video, image


base_folder = "results/5-24-full-server-runs/"
for dataset, data_type in [("FLIC-full", "all"), ("FLIC", "all"), ("FLIC", "test"),
                           ("phoenix", "dev"), ("phoenix", "test"),]:

    # Make flic_data a pandas dataFrame
    flic_data = load_flic_images(data_type, dataset) if "FLIC" in dataset else None
    if "FLIC" in dataset:
        flic_data = pandas.DataFrame([{"video_name": get_video_and_image_name(img_name, dataset)[1],
                                       "image_name": get_video_and_image_name(img_name, dataset)[2],
                      **keyp_dict} for video_list in flic_data.values() for img_name, keyp_dict in video_list])
        flic_data = flic_data.set_index([flic_data["video_name"], flic_data["image_name"]]).sort_index()
        # flic_data.drop(columns="image_name", inplace=True)

    input_filepaths = glob.glob(base_folder + f"run1-{dataset}-{data_type}/" + f"run1-{dataset}-{data_type}" + "*")
    input_filepaths = sorted([filepath for filepath in input_filepaths if ".json" not in filepath and ".csv" not in filepath])

    # grab the no-rule system_outs, for when we need to recompute the diffs with other rules
    no_rule_file = [filepath for filepath in input_filepaths if "no-rule-orig-landmarks" in filepath][0]
    no_rule_table = pandas.read_pickle(no_rule_file)
    no_rule_table[['video_name', 'image_name']] = no_rule_table.apply(lambda row: pandas.Series(get_video_and_image_name(row["input_name"], dataset)[1:]), axis=1)
    no_rule_outputs = pandas.json_normalize(no_rule_table['system_outs'])
    no_rule_outputs.set_index([no_rule_table['video_name'], no_rule_table['image_name']], inplace=True, drop=False)
    no_rule_outputs.sort_index(inplace=True)

    for input_filepath in tqdm(input_filepaths, desc=f"converting files for {dataset}, {data_type}"):
        # if os.path.isfile(input_filepath + "-raw_diffs.csv"):
        #     continue

        table = pandas.read_pickle(input_filepath)
        # pickled_df_columns: "system", "input_name", "met_rules", "eval_diffs", "diff_with_gt", "system_outs"

        # make a new column with only the images filename without the folder path
        table[['video_name', 'image_name']] = table.apply(lambda row: pandas.Series(get_video_and_image_name(row["input_name"], dataset)[1:]), axis=1)
        # sort by image_name to avoid different datasets having a different order
        table.set_index(['video_name', 'image_name'], inplace=True, drop=False, verify_integrity=True)
        table.sort_index(inplace=True)

        # dictionary to save as json with all the extra data that is not the raw keypoints or diffs
        extra_data = {"dataset": dataset, "data_type": data_type, "system_under_test": table.system.iloc[0]}

        # we are currently doing one met_rule per file, this column is just a leftover from earlier iterations
        met_rule_filepath = input_filepath.split(data_type+"-")[1]
        met_rule_table = table.met_rules.iloc[0]
        assert (met_rule_filepath == met_rule_table or met_rule_filepath == met_rule_table.replace("-", "@").replace("_", "-").replace("@", "_")), \
            f"Assertion failed with table_rule={met_rule_table} and input_filepath_split={met_rule_filepath}"
        extra_data["met_rule"] = met_rule_filepath

        # we only need to save the folder path in extra_data, and then only the image number is necessary for each row
        images_folder_path = get_video_and_image_name(table.input_name.iloc[0], dataset)[0]
        assert images_folder_path == get_video_and_image_name(table.input_name.iloc[50], dataset)[0]
        extra_data["images_folder_path"] = images_folder_path

        raw_keypoints = table[["video_name", "image_name", "system_outs"]]
        # split the system_outs into the features that were found.
        # For mediapipe holistic: "left_hand", "right_hand", "face", "pose"
        raw_keypoints = raw_keypoints.join(pandas.json_normalize(raw_keypoints['system_outs']).set_index([raw_keypoints['video_name'], raw_keypoints['image_name']]))
        raw_keypoints.drop(columns="system_outs", inplace=True)

        # accumulate the keypoints and diffs split into columns for each feature here, to concat at the end
        separated_keypoints = []
        separated_diffs_with_no_rule = []

        holistic_features = ["left_hand", "right_hand", "face", "pose"]
        features_num_of_kp = [21, 21, 468, 33]
        for feature, keypoints_length in zip(holistic_features, features_num_of_kp):
            visibility = ["visibility"] if feature == "pose" else []
            keypoints_names = [f'{feature}_{i}_{coord}' for i in range(keypoints_length) for coord in ["x", "y"] + visibility]
            diffs_names = [f'{feature}_{i}' for i in range(keypoints_length)]

            # access the dict with the mediapipe holistic outputs, and split them into columns
            current_separated_keypoints = pandas.DataFrame(raw_keypoints.apply(
                                                lambda row: [None]*len(keypoints_names) if row[f"{feature}"] is None else
                                                           list(itertools.chain.from_iterable(row[f"{feature}"].tolist())), axis=1).tolist(),
                                                        columns=keypoints_names).set_index(raw_keypoints.index)
            current_separated_keypoints[keypoints_names] = current_separated_keypoints[keypoints_names].apply(pandas.to_numeric, downcast="float")
            separated_keypoints.append(current_separated_keypoints)

            # if we are looking at the results of a metRule, save the diffs with the no-rule keypoints.
            if "no-rule-orig-landmarks" not in input_filepath:
                # in the diffs file, save the aggregation for each feature so we know if its nan, and the mean, and each column keypoint
                separated_diffs_with_no_rule.append(pandas.DataFrame(table.apply(lambda row: row["eval_diffs"][feature],
                                                                                 axis=1),
                                                    columns=[feature+"_diff_aggr"]))

                # save the diffs with the original image, so we don't have to recompute them to analyse the data
                all_diffs_one_column = get_non_mean_distance_of_img(no_rule_outputs[feature],
                                                                                 raw_keypoints[feature], feature)
                current_separated_diffs = pandas.DataFrame(all_diffs_one_column.apply(lambda row: mult_cols(row, diffs_names)).tolist(),
                                                                     columns=diffs_names).set_index(no_rule_outputs.index)
                current_separated_diffs[diffs_names] = current_separated_diffs[diffs_names].apply(pandas.to_numeric, downcast="float")
                separated_diffs_with_no_rule.append(current_separated_diffs)

        # add the diff with ground truth if we are in a dataset that has GT
        if "FLIC" in dataset:
            gt_keypoints = [f'gt_{keyp}' for keyp in flic_data.columns if "image_name" not in keyp and "video_name" not in keyp]
            separated_diffs_with_no_rule.append(
                pandas.DataFrame(table.apply(lambda row: row["diff_with_gt"], axis=1), columns=["gt_diff_aggr"]))
            separated_gt_diffs = []
            for img_idx in flic_data.index:
                diffs_for_img = diff_holistic_pose_and_flic_ground_truth(flic_data.loc[img_idx],
                                                                         raw_keypoints.loc[img_idx]["pose"])
                separated_gt_diffs.append(mult_cols(diffs_for_img, gt_keypoints))
            separated_gt_diffs = pandas.DataFrame(separated_gt_diffs, columns=gt_keypoints).set_index(raw_keypoints.index)
            separated_gt_diffs[gt_keypoints] = separated_gt_diffs[gt_keypoints].apply(pandas.to_numeric, downcast="float")
            separated_diffs_with_no_rule.append(separated_gt_diffs)
        else:
            # if the dataset doesn't have ground truth annotations, don't save these columns
            pass

        finished_keypoints = pandas.concat([raw_keypoints] + separated_keypoints, axis=1)
        finished_keypoints.drop(columns=holistic_features, inplace=True)
        finished_keypoints.sort_index(inplace=True)
        finished_keypoints.reset_index(drop=True, inplace=True)
        finished_keypoints['video_name'] = finished_keypoints['video_name'].astype("category")
        # raw_keypoints are ready to save now, a csv of only the filename and all the columns keypoints.
        finished_keypoints.to_csv(input_filepath+"-raw_keypoints.csv")

        finished_diffs = pandas.concat([raw_keypoints] + separated_diffs_with_no_rule, axis=1)
        finished_diffs.drop(columns=holistic_features, inplace=True)
        finished_diffs.sort_index(inplace=True)
        finished_diffs.reset_index(drop=True, inplace=True)
        finished_diffs['video_name'] = finished_diffs['video_name'].astype("category")
        # raw_diffs are ready to save now, a csv of only the filename and all the columns of diffs for each keypoint.
        finished_diffs.to_csv(input_filepath + "-raw_diffs.csv")

        # in the aggregations file we save: imgs_folder_path, met_rule, dataset, data_type, system under test, etc.
        with open(input_filepath + "-extra_data.json", "w") as write_file:
            json.dump(extra_data, write_file)
