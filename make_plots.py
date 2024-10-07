# Import packages
import argparse
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


# Possible implemented settings, can be expanded======
possible_features = ["left_hand", "right_hand", "face", "pose", "gt"]
possible_image_aggregations = ["mean", "median", "max", "min"]
possible_cross_img_aggregations = ["table_subsumption_ratios", "histogram_failed_images_across_thresholds",
                                   "gt_vs_metamorphic_failed_tests_across_thresholds", "images_failed_per_num_of_rules"]

#=======
def find_all_rules_results(base_results_folder, dataset):
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
    #all_possible_rules = [rule for group in coarse_possible_rules for rule in coarse_possible_rules[group]]
    return coarse_possible_rules
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

    cache_folder = "./Results/plots_data/"

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
                data[i][j] = 1  # if rule_1 never finds any faults, it subsumes completely
            else:
                data[i][j] = errors_in_common / num_errors_1

    pretty_rules_names = [rule.replace('img-', '').replace('masked_', '').replace('color-', '').replace('dark-bright', 'brightness') for rule in rules]
    pretty_rules_names = ['brightness_gamma_'+rule.split('_')[-2] if "gamma" in rule else rule for rule in pretty_rules_names]
    pretty_rules_names = ['brightness_scale_'+'_'.join(rule.split('_')[1:3]) if "scale" in rule else rule for rule in pretty_rules_names]
    fig = px.imshow(data, labels=dict(x="subsumed rule", y="subsuming rule", color="subsumption rate"),
                    x=pretty_rules_names, y=pretty_rules_names, text_auto=True)
    fig.update_traces(texttemplate='%{text:.2f}')
    fig.update_xaxes(side="top", tickangle=45)
    fig.update_yaxes(tickangle=45)
    fig.update_layout(yaxis_nticks=len(rules), xaxis_nticks=len(rules), width=1400, height=1400, font_size=16,
                      legend=dict(itemsizing="constant"),
                      )
    data = pd.DataFrame(data)
    return fig, data


def select_final_rules(rules_to_plot, plot_all_rules, all_possible_rules, all_motion_blur=False, allRulesMinusIdent=False):
    if plot_all_rules:
        rules_to_plot = [[rule for rule in all_possible_rules[group]] for group in all_possible_rules.keys()]
    elif allRulesMinusIdent:
        rules_to_plot = [[rule for rule in all_possible_rules[group] if "identity" not in rule and "orig" not in rule]
                         for group in all_possible_rules.keys()]
    elif all_motion_blur:
        rules_to_plot = [all_possible_rules["img-motion-blur"]]
    else:
        for i, rules_list in enumerate(rules_to_plot):
            all_rules = any(["all_rules_in_" in specific_rule for specific_rule in rules_list])
            rules_list = [] if (len(rules_list)==0 or "none" in rules_list) else (all_possible_rules[rules_list[0].split('_')[-1]] if all_rules else [rules_list])
            rules_to_plot[i] = rules_list

    return rules_to_plot

def save_modified_images(rules_selected_to_plot, dataset, base_images_folder):

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
                                'dataset_name': dataset.split('-')[-2]}
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
                  base_results_folder, range_thresholds, rules_codename):

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
        # print(f"doing {feature}")
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


    # print("full dataset to plot")
    # print(all_plot_data)

    latex_table = None

    list_rules_to_plot = [rule for rule_coarse_group in rules_selected_to_plot for rule in rule_coarse_group]

    # here start the different plotting options
    if plot_type == "table_subsumption_ratios":
        fig, plotted_dataframe = table_subsumption(all_plot_data, list_rules_to_plot, keypoint_type, aggregation_metric, error_threshold)

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

        # print("histogram_df:")
        # print(histogram_df)

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

        # print("table_df:")
        # print(table_df)

        plotted_dataframe = table_df
        fig = None
        latex_table = table_df.to_latex(index_names=False, index=False)


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

        if plot_type == "histogram_failed_images_across_thresholds":
            histogram_df.drop("failed_images", inplace=True, axis=1)
        # print("final histogram df to plot")
        # print(histogram_df)
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

    plots_folder = "./plots"
    pathlib.Path(plots_folder).mkdir(parents=True, exist_ok=True)

    if fig is not None:
        fig.update_layout(
            title={'text': dataset.split('-')[1], 'y': 0.93},
            font_size=22
        )
        fig.write_image(f"{plots_folder}/{plot_type}_{keypoint_type}_{dataset}_{rules_codename}.pdf")

    print(f"plotted dataframe: \n {plotted_dataframe}")
    plotted_dataframe.to_csv(f"{plots_folder}/{plot_type}_{keypoint_type}_{dataset}_{rules_codename}.csv")

    if latex_table is not None:
        with open(f"{plots_folder}/{plot_type}_{keypoint_type}_{dataset}_{rules_codename}_latex.txt", "w") as file:
            file.write(latex_table)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-res_dir', '--base_results_folder', type=str, default='./Results/',  # required=True,
                        help='Path to the folder with the csv files. If no value is provided "./Results/" is used as'
                             'default.')
    parser.add_argument('-data_dir', '--base_images_folder', type=str, default='./Datasets/',  # required=True,
                        help='Path to the folder with the csv files. If no value is provided "./Datasets/" is used as'
                             'default.')
    parser.add_argument('-dataset', '--dataset', type=str, required=True,
                        help='What dataset inside of the data_dir folder to plot. Can be FLIC-test, phoenix-dev, etc.')
    parser.add_argument('-kp_type', '--keypoints_type', action='append', type=str, required=True, choices=possible_features,
                        help=f'What type of keypoints to aggregate, can be more than one. From {possible_features}.'
                             f' gt is for ground truth, and needs to be included when comparing'
                             'the metamorphic testing results to ground truth based results')
    parser.add_argument('-img_aggr', '--kp_aggregation_metric', type=str, default='median', choices=possible_image_aggregations,
                        help=f'How to aggregate keypoints errors in an image to one value. Can be {possible_image_aggregations}'
                             'If not value is provided, default is "median"')
    parser.add_argument('-plot_type', '--plot_type', type=str, choices=possible_cross_img_aggregations,
                        help='How to aggregate keypoints errors in an image to one value. Can be one of '
                             f'{possible_cross_img_aggregations}.'
                             f'"histogram_failed_images_across_thresholds" was used for RQ1, '
                              '"gt_vs_metamorphic_failed_tests_across_thresholds", was used for RQ2,'
                             f'"table_subsumption_ratios" was used for RQ3 subsumption heatmaps. '
                             ' "images_failed_per_num_of_rules" was used for RQ3 table')
    parser.add_argument('-err_thresh', '--error_threshold', default=0.2,
                        help='What error threshold to use for RQ3 subsumption heatmaps and number of images failed per'
                             'number of metamorphic rules. If not provided, default value is 0.2. Ignored for the other'
                             'plot types')
    parser.add_argument('-rg_errs_threshs', '--range_errors_thresholds', type=str,
                        default="[0.01, 0.05, 0.1, 0.2, 0.3, 0.7, 1, 2, 5, 20, 9999999]",
                        help='List of values of error thresholds to plot for RQ1 and RQ2 bar plots.'
                             'If no value is provided "[0.01, 0.05, 0.1, 0.2, 0.3, 0.7, 1, 2, 5, 20, 9999999]"'
                             'is used as default.'
                             '9999999 or any very high value is the stand in for "inf" type errors')
    parser.add_argument('-save_mod_imgs', '--save_modified_images', action="store_true",
                        help='If flag is set, saves in "./modified_images" some examples of images of the dataset'
                             'after being modified by all the metamorphic transformations listed as input to this script'
                             'Default to False')
    parser.add_argument('-metR', '--metamorphic_rules', type=str, required=True,
                        help='List of metamorphic rules to take into account when plotting how many images failed'
                             )
    parser.add_argument('-rules_codename', type=str, required=True,
                        help='Name to easily recognize saved results'
                        )

    # Check which metRules are in the program arguments and parse their relevant kwargs
    recognized_args, unrecognized_args = parser.parse_known_args()

    base_results_folder = recognized_args.base_results_folder
    base_images_folder = recognized_args.base_images_folder
    dataset = recognized_args.dataset  # 'FLIC-test', 'phoenix-dev', etc.
    kp_type = recognized_args.keypoints_type  # ["pose", "gt"]
    img_aggr = recognized_args.kp_aggregation_metric
    plot_type = recognized_args.plot_type
    error_threshold = recognized_args.error_threshold
    range_error_thresholds = recognized_args.range_errors_thresholds
    save_modified_images_flag = recognized_args.save_modified_images
    input_rules_to_plot = recognized_args.metamorphic_rules
    rules_codename = recognized_args.rules_codename

    coarse_possible_rules = find_all_rules_results(base_results_folder, dataset)
    if input_rules_to_plot == "AllRels":
        rules_selected_to_plot = select_final_rules([], True, coarse_possible_rules)
    elif input_rules_to_plot == "FailedPerNumRulesTable":
        rules_selected_to_plot = select_final_rules([], False, coarse_possible_rules, allRulesMinusIdent=True)
    elif input_rules_to_plot == "all_rules_in_img-motion-blur":
        rules_selected_to_plot = select_final_rules([], False, coarse_possible_rules, True)
    else:
        input_rules_to_plot = input_rules_to_plot.replace("(0.5, 0.5)", "(0.5: 0.5)")
        input_rules_to_plot = input_rules_to_plot.split(",")
        input_rules_to_plot = [rule.replace("(0.5: 0.5)", "(0.5, 0.5)") for rule in input_rules_to_plot]
        input_rules_to_plot = [rule.strip("'") for rule in input_rules_to_plot]
        rules_selected_to_plot = select_final_rules(input_rules_to_plot, False, coarse_possible_rules)

    print(f"doing rules {rules_codename}, dataset {dataset}, kp_type {kp_type}, plot type {plot_type}")

    compute_graph(rules_selected_to_plot, kp_type, img_aggr, plot_type, dataset, error_threshold,
                  base_results_folder, range_error_thresholds, rules_codename)

    if save_modified_images_flag:
        save_modified_images(rules_selected_to_plot, dataset, base_images_folder)
