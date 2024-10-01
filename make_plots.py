# Import packages
import glob
import math
import os
import pathlib
import random
import re

import cv2
import numpy as np
import pandas
import pandas as pd
import plotly.express as px

from metamorphic_rules import instantiate_rules


# Settings that can be changed======
possible_features = ["left_hand", "right_hand", "face", "pose", "gt"]
possible_image_aggregations = ["mean", "median", "max", "min"]
possible_cross_img_aggregations = ["table_subsumption_ratios", "histogram_failed_images_across_thresholds",
                                   "gt_vs_metamorphic_failed_tests_across_thresholds", "images_failed_per_num_of_rules"]
SubRels = ["identity", "img-black-white", "no-rule-orig-landmarks",
             "img-blur_10_3", "img-blur_125_5", "img-blur_80_7",
             "img-dark-bright_20_0.8_1.2_scale", "img-dark-bright_20_1.2_0.5_gamma",
             "img-mirror_1",
             "img-motion-blur_11_0", "img-motion-blur_11_100", "img-resolution_0.2",
             "img-resolution_0.7", "img-rotation_5_(0.5, 0.5)", "img-rotation_10_(0.5, 0.5)",
             "img-stretch_1.25_1", "img-stretch_1_0.6", "img-stretch_1_0.8",
             "img-masked_background_img-color-wheel_90", "img-masked_hair_img-color-wheel_90",]

base_results_folder = './Results/'

base_images_folder = './Datasets'

dataset = 'FLIC-test'  # phoenix-dev

results_selection = ["pose", "gt"]  # l_hand, r_hand, face

scores_selection = "median"  # mean, max, min

column_selection = "histogram_failed_images_across_thresholds"

subsumption_error_threshold = 0.2

range_error_thresholds = "[0.01, 0.05, 0.1, 0.2, 0.3, 0.7, 1, 2, 5, 20, 9999999]"

build_transformed_images = False
#=======
# script to put down in main


def natural_sort(l):
    # https: // stackoverflow.com / questions / 11150239 / natural - sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Find all rules ran in experiment, by finding raw_diffs files
input_filepaths = glob.glob(base_results_folder + f"{dataset}/" + f"{dataset}" + "*")
all_possible_rules = natural_sort([filename.rpartition(f"{dataset}-")[2].rpartition("-raw_diffs")[0]
                              for filename in input_filepaths if "raw_diffs" in filename])


# organize the rules in coarse, which group the different settings that we try for example for blur
coarse_possible_rules = sorted(list(set([rule.split('_')[0] for rule in all_possible_rules])))
coarse_possible_rules = {rule_group: [rule for rule in all_possible_rules if rule_group in rule]
                         for rule_group in coarse_possible_rules}
all_possible_rules = [rule for group in coarse_possible_rules for rule in coarse_possible_rules[group]]
#===========================


def per_img_aggr(results_series, aggr_str):
    for special_val in [None, math.nan, math.inf, -math.inf]:
        # for each mediapipe feature, either no keypoints are returned, or all are returned.
        if (results_series == special_val).any():
            assert (results_series == special_val).all()
    if aggr_str == "mean":
        return results_series.mean()
    elif aggr_str == "median":
        return results_series.median()
    elif aggr_str == "max":
        return results_series.max()
    elif aggr_str == "min":
        return results_series.min()
    else:
        # when adding new per_img_aggr, check how it works if the series has nans, inf, -inf
        raise (ValueError, f"please implement how to do this image aggregation {aggr_str}")


def compute_requested_data(base_folder, dataset_name, feature, img_aggr, selected_rule_group, already_computed_columns):

    answer_data = pandas.DataFrame(columns=[f"{img_aggr}-{selected_rule}" for selected_rule in selected_rule_group])

    for selected_rule in selected_rule_group:
        print(f"starting update for {selected_rule}")
        if f"{img_aggr}-{selected_rule}" in already_computed_columns:
            continue  # this settings combination was already computed

        # Load the csv files for the new selected rule and save their aggregations to be plotted
        current_rule_df = pd.read_csv(
            os.path.join(f"{base_folder}{dataset_name}/{dataset_name}-{selected_rule}-raw_diffs.csv"))

        # asure that different dfs have the same order, for when saving the aggregations per image
        current_rule_df.sort_values(["video_name", "image_name"], inplace=True)
        # save the video and image name in the summary.csv
        current_rule_df.set_index(["video_name", "image_name"], inplace=True)

        # keep only the columns for the current feature
        cols_of_current_feat = [col for col in current_rule_df.columns if feature in col]
        current_feat_df = current_rule_df[cols_of_current_feat]

        # aggregate the values of the different keypoints for each image
        current_results = current_feat_df.apply(lambda row: per_img_aggr(row, img_aggr), axis=1)

        answer_data[f"{img_aggr}-{selected_rule}"] = current_results
    return answer_data


def load_update_data_from_disk(base_folder, dataset_name, feature, img_aggr, selected_rule_group):
    """ if file exists, load csv dataframe. If requested img_aggr exist, return it.
     Otherwise compute it and add value/column and save the csv. """

    cache_folder = "./results/plots_data/"

    if not os.path.isdir(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)

    rule_group_name = selected_rule_group[0].split('_')[0]
    for rule in selected_rule_group:
        assert rule.split('_')[0] == rule_group_name

    # we have one file per dataset, feature and metRel_group
    relevant_filepath = cache_folder + f"{dataset_name}-{feature}-{rule_group_name}-plots-summary.csv"

    if os.path.isfile(relevant_filepath):
        read_data = pd.read_csv(relevant_filepath)
        read_data.set_index(["video_name", "image_name"], inplace=True)
        try:
            answer_data = read_data[[f"{img_aggr}-{selected_rule}" for selected_rule in selected_rule_group]]
        except KeyError:
            already_computed_columns = read_data.columns.values
            answer_data = compute_requested_data(base_folder, dataset_name, feature, img_aggr, selected_rule_group,
                                                 already_computed_columns)
            data_to_save = pd.concat([read_data, answer_data], axis=1)
            data_to_save.to_csv(relevant_filepath)
    else:
        answer_data = compute_requested_data(base_folder, dataset_name, feature, img_aggr, selected_rule_group, [])
        answer_data.to_csv(relevant_filepath)
    return answer_data


def did_image_fail(error_diff, error_threshold):
    if isinstance(error_diff, str):
        raise ValueError(f"{error_diff} is supposed to be the error number but it is a string")
    # takes only one error float, as in, the aggregated error of one image
    if error_diff is None or math.isnan(error_diff):
        return False
    elif error_diff == math.inf:
        # if the model identifies this feature in the modified image but not in the original img
        return True
    elif error_diff == -math.inf:
        # if the model identifies this feature in the original image but not in the modified img
        return True
    else:
        return error_diff >= error_threshold


def table_subsumption(all_plot_data, rules, features, img_aggr, error_threshold):
    # Only to be called by update_graph. returns fig, a heatmap of subsumption rates between rules,
    # depending on the error_threshold

    if len(features) > 1:
        print("Please choose only one feature (l_hand, face, pose) when plotting table_subsumption")
        raise ValueError("Please choose only one feature (l_hand, face, pose) when plotting table_subsumption")

    relevant_results = all_plot_data[features[0]]

    print("started building subsumption table")
    imgs_failed_per_rule = {rule: [did_image_fail(error_diff, error_threshold)
                                       for error_diff in relevant_results[f"{img_aggr}-{rule}"]
                                       ]
                                for rule in rules}
    print("Done counting errors per rule")

    data = np.empty(shape=(len(rules), len(rules))).tolist()
    for i, rule_1 in enumerate(rules):
        errors_per_img_rule_1 = imgs_failed_per_rule[rule_1]
        num_errors_1 = sum(errors_per_img_rule_1)
        for j, rule_2 in enumerate(rules):
            errors_per_img_rule_2 = imgs_failed_per_rule[rule_2]
            errors_in_common = sum([failed_1 and failed_2
                                    for failed_1, failed_2 in zip(errors_per_img_rule_1, errors_per_img_rule_2)])

            if num_errors_1 == 0:
                data[i][j] = 1  # if rule_1 never finds any faults, it is completely subsumed
            else:
                data[i][j] = errors_in_common / num_errors_1

    pretty_rules_names = [rule.replace('img-', '').replace('masked_', '').replace('color-', '').replace('dark-bright', 'brightness') for rule in rules]
    pretty_rules_names = ['brightness_gamma_'+rule.split('_')[-2] if "gamma" in rule else rule for rule in pretty_rules_names]
    pretty_rules_names = ['brightness_scale_'+'_'.join(rule.split('_')[1:3]) if "scale" in rule else rule for rule in pretty_rules_names]
    fig = px.imshow(data, labels=dict(x="subsuming rule", y="subsumed rule", color="subsumption rate"),
                    x=pretty_rules_names, y=pretty_rules_names, text_auto=True)
    fig.update_traces(texttemplate='%{text:.2f}')
    fig.update_xaxes(side="top", tickangle=45)
    fig.update_yaxes(tickangle=45)
    fig.update_layout(yaxis_nticks=len(rules), xaxis_nticks=len(rules), width=1400, height=1400, font_size=16,
                      legend=dict(itemsizing="constant"),
                      )
    data = pd.DataFrame(data)
    return fig, data


def select_final_rules(rules_to_plot, plot_all_rules, all_possible_rules):
    if plot_all_rules:
        rules_to_plot = [[rule for rule in all_possible_rules[group]] for group in all_possible_rules.keys()]

    for i, rules_list in enumerate(rules_to_plot):
        all_rules = any(["all_rules_in_" in specific_rule for specific_rule in rules_list])
        rules_list = [] if (len(rules_list)==0 or "none" in rules_list) else (all_possible_rules[rules_list[0].split('_')[-1]] if all_rules else rules_list)
        rules_to_plot[i] = rules_list

    return rules_to_plot

# TODO rules_selected_to_plot = [input list] or "all" -> all_possible_rules or "subRels" -> subRels? or "all from X group"
rules_selected_to_plot = select_final_rules(input_selected_rules)


def show_modified_images(rules_selected_to_plot, dataset, base_images_folder):

    original_images_paths = []
    images_to_show = []
    if not os.path.isdir("./modified_images"):
        os.makedirs("./modified_images", exist_ok=True)

    if "FLIC" in dataset:
        base_images_folder = f"{base_images_folder}/FLIC/FLIC{'-full' if 'full' in dataset else ''}/images/"
        original_images_paths = os.listdir(base_images_folder)
        selected_imgs = random.sample(original_images_paths, 3)
        original_images_paths = [base_images_folder + img for img in selected_imgs]
    elif "phoenix" in dataset:
        base_images_folder = f"{base_images_folder}/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{dataset.split('-')[-1]}"
        videos_list = os.listdir(base_images_folder)
        videos_list = random.sample(videos_list, 3)
        for video in videos_list:
            images = os.listdir(base_images_folder+"/"+video)
            selected_image = random.sample(images, 1)[0]
            original_images_paths.append(base_images_folder+"/"+video+"/" + selected_image)
    else:
        raise ValueError("dataset not found, check code or files")

    print("images_paths")
    print(original_images_paths)

    list_rules_to_plot = [rule for rule_coarse_group in rules_selected_to_plot for rule in rule_coarse_group]
    for full_image_path in original_images_paths:
        for rule in list_rules_to_plot:
            if rule == "no-rule-orig-landmarks":
                images_to_show.append(full_image_path)
            else:
                orig_image = cv2.imread(full_image_path)
                # Convert the BGR image from opencv to RGB, we need to modify the image in the same way as when we
                # computed the mediapipe keypoints
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

                met_rule_instance = instantiate_rules.met_rule_result_str_to_instance(rule)
                extra_kwargs = {'image_filepath': full_image_path,
                                'save_modified_images': False,
                                'data_type': dataset.split('-')[-1],
                                'dataset_name': dataset.split('-')[1]}
                modified_image = met_rule_instance.apply(orig_image, extra_kwargs)
                cv2.putText(modified_image, rule, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, 2)

                current_modif_img_path = f"./modified_images/{len(os.listdir('./modified_images/'))}.jpg"
                # Convert RGB image to BGR channel order so cv2 can save it
                modified_image = cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR)
                # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
                cv2.imwrite(current_modif_img_path, modified_image)
                images_to_show.append(current_modif_img_path)

    print(f"finished creating the modified images, in total {len(images_to_show)} imgs")


def compute_graph(rules_selected_to_plot, keypoint_type, aggregation_metric, plot_type, dataset, error_threshold,
                  base_results_folder, range_thresholds, build_images_bool):

    if build_images_bool:
        show_modified_images(rules_selected_to_plot, dataset, base_images_folder)

    threshold_buckets = list(map(float, range_thresholds.strip("[]").split(", ")))
    threshold_buckets = [(-0.0000001, threshold_buckets[i]) if i == 0 else
                         (threshold_buckets[i - 1] - 0.0000001, threshold_buckets[i]) for i in
                         range(len(threshold_buckets))]
    plot_thresholds = [str(item[1]) if item[1] < 99999 else "inf" for item in threshold_buckets]

    # load or compute the values needed for the requested plot
    # all_plot_data has a row for evey image in the dataset. As many columns as combinations of {img_aggr}-{met_rule}
    # have been saved, and the video and image name to be used for indexing.
    all_plot_data = {feature: pandas.DataFrame() for feature in keypoint_type}
    for feature in keypoint_type:
        print(f"doing {feature}")
        for rule_group in rules_selected_to_plot:
            if len(rule_group) == 0:
                continue

            # for ground truth, we mostly want to plot the original image no-rule diff with gt, not the follow ups modified imgs
            if feature == "gt"\
                    and (len(rule_group) != 1 or rule_group[0] != "no-rule-orig-landmarks"):
                continue

            requested_data = load_update_data_from_disk(base_results_folder, dataset, feature, aggregation_metric, rule_group)
            all_plot_data[feature] = pd.concat([all_plot_data[feature], requested_data], axis=1)

        if feature == "gt" and len(all_plot_data["gt"].columns) == 0:
            print("make sure that no-rule-orig-landmarks is selected in the metamorphic rules to plot gt")
            raise ValueError(
                "make sure that no-rule-orig-landmarks is selected in the metamorphic rules to plot gt")


    print("full dataset to plot")
    print(all_plot_data)


    list_rules_to_plot = [rule for rule_coarse_group in rules_selected_to_plot for rule in rule_coarse_group]

    # here start the different plotting options
    if plot_type == "table_subsumption_ratios":
        fig, plotted_dataframe = table_subsumption(all_plot_data, list_rules_to_plot, keypoint_type, aggregation_metric, error_threshold)
        fig.show()

    elif plot_type == "images_failed_per_num_of_rules":
        if len(keypoint_type) > 1:
            print("Please choose only one feature (l_hand, face, pose) when plotting num failed for threshold")
            raise ValueError("Choose only one feature (l_hand, face, pose) when plotting num failed for threshold")

        histogram_df = pandas.DataFrame()
        for feature in keypoint_type:
            # we replace -inf with inf, since they both represent the maximum error
            all_plot_data[feature].replace(-math.inf, math.inf, inplace=True)

            current_df = all_plot_data[feature]
            for min_err, max_err in threshold_buckets:
                num_failed_rules = current_df.apply(
                    lambda row: (row > max_err).sum(), axis=1
                )
                errors_df = pd.DataFrame({"num_failed_rules": num_failed_rules,
                                          "feature": feature,
                                          "plot_thresholds": max_err})

                histogram_df = pandas.concat([histogram_df, errors_df])

        print("histogram_df:")
        print(histogram_df)

        def imgs_failed_per_num_rules(how_many_rules_we_count, rules_failed_per_img):
            if how_many_rules_we_count == "total images":
                return len(rules_failed_per_img)
            elif "fails all rules" in how_many_rules_we_count:
                how_many_rules_we_count = (len(list_rules_to_plot), len(list_rules_to_plot))
            elif "more than three quarters" in how_many_rules_we_count:
                how_many_rules_we_count = (int(len(list_rules_to_plot) * 3 / 4), len(list_rules_to_plot) -1)
            elif "between half" in how_many_rules_we_count:
                how_many_rules_we_count = (int(len(list_rules_to_plot) / 2), int(len(list_rules_to_plot) * 3 / 4))
            elif "to half" in how_many_rules_we_count:
                how_many_rules_we_count = (int(len(list_rules_to_plot) / 4), int(len(list_rules_to_plot) / 2))
            elif "to 25%" in how_many_rules_we_count:
                how_many_rules_we_count = (5, int(len(list_rules_to_plot) / 4))
            elif "-" in how_many_rules_we_count:
                how_many_rules_we_count = (int(how_many_rules_we_count.split("-")[0]), int(how_many_rules_we_count.split("-")[1]))
            else:
                how_many_rules_we_count = (int(how_many_rules_we_count), int(how_many_rules_we_count))
            ans = ((rules_failed_per_img >= how_many_rules_we_count[0]) & (rules_failed_per_img <= how_many_rules_we_count[1])).sum()
            return ans

        ranges_of_number_of_rules_failed = ["0", "1", "2", "3", "4", f"5 to 25% ({int(len(list_rules_to_plot)/4)})", f"25% to half ({int(len(list_rules_to_plot)/2)}) the rules", f"between half and three quarters ({int(len(list_rules_to_plot)*3/4)})", "more than three quarters but not all", f"fails all rules ({int(len(list_rules_to_plot))})", f"total images"]
        table_df = pd.DataFrame({"num_failed_rules": pd.Series(ranges_of_number_of_rules_failed)})
        for _, max_err in threshold_buckets:
            current_error_failed_images_series = histogram_df[(histogram_df["plot_thresholds"] == max_err)]["num_failed_rules"]
            table_df.insert(len(table_df.columns), f"{max_err}",
                            pd.Series([imgs_failed_per_num_rules(num_rules, current_error_failed_images_series) for num_rules in ranges_of_number_of_rules_failed]))

        print("table_df:")
        print(table_df)

        latex_table = table_df.to_latex(index_names=False, index=False)
        # return no fig plot but the table to the div children
        return latex_table

    elif (plot_type == "histogram_failed_images_across_thresholds"
          or plot_type == "gt_vs_metamorphic_failed_tests_across_thresholds"):

        original_features = keypoint_type
        if plot_type == "gt_vs_metamorphic_failed_tests_across_thresholds":
            if "gt" not in keypoint_type:
                print("gt has to be in the selected features to plot gt_vs_metamorphic_failed_tests_across_thresholds")
                raise ValueError(
                    "gt has to be in the selected features to plot gt_vs_metamorphic_failed_tests_across_thresholds")
            keypoint_type = ["gt", "metamorphic"]

        histogram_df = pandas.DataFrame()
        for feature in keypoint_type:
            if feature == "metamorphic":
                # combine the highest error of all the metamorphic features
                current_df = pd.DataFrame()
                for met_feature in original_features:
                    if met_feature == "gt":
                        continue
                    # we replace -inf with inf, since they both represent the maximum error
                    all_plot_data[met_feature].replace(-math.inf, math.inf, inplace=True)

                    inside_df = all_plot_data[met_feature]
                    current_df[met_feature] = inside_df.apply(
                        lambda row: row.max(), axis=1
                        # grab the largest error across all requested metamorphic relations
                        # (so the biggest error in the follow-up test set for each test/image)
                    )
                    current_df["max_errors"] = current_df.apply(max, axis=1)
            else:
                # we replace -inf with inf, since they both represent the maximum error
                all_plot_data[feature].replace(-math.inf, math.inf, inplace=True)

                current_df = all_plot_data[feature]
                current_df["max_errors"] = current_df.apply(
                    lambda row: row.max(), axis=1
                    # grab the largest error across all requested metamorphic relations
                    # (so the biggest error in the follow-up test set for each test/image)
                )

            # we are not counting only the numbers, but also the infs, but not the nans.
            failed_images = [(current_df["max_errors"] >= max_err)
                            for (min_err, max_err) in threshold_buckets]

            # just for diagnostics, can be removed
            nans = (current_df["max_errors"].isna()).sum()
            inf = (current_df["max_errors"] == math.inf).sum()
            neg_inf = (current_df["max_errors"] == -math.inf).sum()
            print(f"for {feature}:")
            print(f"number of found nans: {nans}; inf: {inf}, -inf: {neg_inf} out of total of: {len(current_df)}")
            print(f"percentage of nans: {nans/len(current_df)}; inf: {inf/len(current_df)}, -inf: {neg_inf/len(current_df)}")

            errors_df = pd.DataFrame({"failed_images": pandas.Series(failed_images),
                                      "feature": feature,
                                      "plot_thresholds": plot_thresholds})
            histogram_df = pandas.concat([histogram_df, errors_df])

        if plot_type == "gt_vs_metamorphic_failed_tests_across_thresholds":
            # we want to plot the number of images that fail for both, only for gt, and only for met.
            # Not how many fail for gt and metamorphic independently
            met_failed_per_thresh = histogram_df[histogram_df["feature"] == "metamorphic"]["failed_images"]
            gt_failed_per_thresh = histogram_df[histogram_df["feature"] == "gt"]["failed_images"]

            both_failed_per_thresh = [[(met and gt) for met, gt in zip(met_failed_per_thresh[idx], gt_failed_per_thresh[idx])]
                                      for idx in range(len(plot_thresholds))]
            met_failed_per_thresh = [[(met and not both) for met, both in zip(met_failed_per_thresh[idx], both_failed_per_thresh[idx])]
                                      for idx in range(len(plot_thresholds))]
            gt_failed_per_thresh = [[(gt and not both) for gt, both in zip(gt_failed_per_thresh[idx], both_failed_per_thresh[idx])]
                                    for idx in range(len(plot_thresholds))]

            both = pd.DataFrame({"error_counts": [sum(failed)/len(failed) for failed in both_failed_per_thresh],
                                 "feature": "both failed",
                                 "plot_thresholds": plot_thresholds})
            met = pd.DataFrame({"error_counts": [sum(failed)/len(failed) for failed in met_failed_per_thresh],
                                 "feature": "only metamorphic",
                                 "plot_thresholds": plot_thresholds})
            gt = pd.DataFrame({"error_counts": [sum(failed)/len(failed) for failed in gt_failed_per_thresh],
                                "feature": "only ground truth",
                                "plot_thresholds": plot_thresholds})

            histogram_df = pandas.concat([both, met, gt])
        else:
            assert plot_type == "histogram_failed_images_across_thresholds"
            # normalise to get percentage instead of the count
            histogram_df["error_counts"] = histogram_df.apply(
                lambda row: (row["failed_images"].sum()) / len(row["failed_images"]), axis=1
            )

        print("final histogram df to plot")
        print(histogram_df)
        plotted_dataframe = histogram_df
        fig = px.bar(histogram_df, x="plot_thresholds", y="error_counts", text="error_counts",
                     pattern_shape="feature", color="feature", pattern_shape_sequence=["", "+", "/"],
                     labels={'plot_thresholds': 'normalised error threshold', 'error_counts': 'percentage of failing tests',
                             "feature": 'approach counted' if plot_type == "gt_vs_metamorphic_failed_tests_across_thresholds" else 'key-points type'}, )
        fig.update_traces(textposition='auto', textangle=0, texttemplate='%{text:.4f}')
        fig.update_layout(legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1,
                                      xanchor="right",
                                      x=1),
                          uniformtext_minsize=10, uniformtext_mode='show')

    else:
        raise ValueError(f"Need to implement this plot {plot_type}")

    fig.update_layout(
        title={'text': dataset.split('-')[1], 'y': 0.93},
        font_size=22
    )
    fig.show()

    print(f"plotted_dataframe: \n {plotted_dataframe}")
    rules_file_name = list_rules_to_plot if len(list_rules_to_plot) < 4 else len(list_rules_to_plot)
    plots_csvs_folder = "./plots/plots_csvs"
    pathlib.Path(plots_csvs_folder).mkdir(parents=True, exist_ok=True)
    plotted_dataframe.to_csv(f"{plots_csvs_folder}/{plot_type}_{keypoint_type}_{dataset}_rulesLen{rules_file_name}.csv")
    return fig



if __name__ == '__main__':

