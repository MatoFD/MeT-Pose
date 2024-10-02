import math
import multiprocessing
import os
import pathlib
import cv2
import mediapipe as mp
import pandas
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

from collections import OrderedDict
import glob

from metamorphic_rules import instantiate_rules
from evaluation_metrics.poses_evaluation import get_2d_landmark_diffs, pck3d, normalized_mean_absolute_error
from test_systems import aux_functions


def holistic_metric(orig_landmark, modif_landmark, metric_type, scoring_method="2d_dist"):
    if scoring_method == "pck3d":
        return pck3d(orig_landmark, modif_landmark)
    elif scoring_method == "2d_dist":
        return normalized_mean_absolute_error(orig_landmark, modif_landmark, metric_type)
    else:
        raise ValueError(f"scoring method not recognized, got {scoring_method}")


def diff_holistic_with_ground_truth(gt, holistic, dataset_name):
    if dataset_name == "FLIC" or dataset_name == "FLIC-full":
        return mean_diff_holistic_pose_and_flic_gt(gt, holistic)
    elif dataset_name == "phoenix":
        return None
    else:
        raise "If using a new dataset, please implement how to compare against that dataset ground truth"


def mean_diff_holistic_pose_and_flic_gt(flic_gt, mediapipe_landmarks, scoring_method="2d_dist"):
    diffs = diff_holistic_pose_and_flic_ground_truth(flic_gt, mediapipe_landmarks, scoring_method=scoring_method)
    if scoring_method == "pck3d":
        return diffs  # it's already aggregated as the num of correct keypoints
    else:
        if (not isinstance(diffs, np.ndarray)) and (diffs == math.inf or diffs == -math.inf):
            return diffs
        else:
            return np.sum(diffs) / len(diffs)


def diff_holistic_pose_and_flic_gt_all_imgs(flic_gt, mediapipe_landmarks, scoring_method="2d_dist"):
    # get the diffs for each keypoint for all the images
    assert len(flic_gt) == len(mediapipe_landmarks)
    ans = {}
    for filename in flic_gt.index:
        ans[filename] = diff_holistic_pose_and_flic_ground_truth(flic_gt.loc[filename],
                                                                 mediapipe_landmarks.loc[filename],
                                                                 scoring_method)
    return pandas.DataFrame(ans)


def diff_holistic_pose_and_flic_ground_truth(flic_gt, mediapipe_landmarks, scoring_method="2d_dist"):
    # flic only has 11-12 landmarks, that we will compare against only the pose component of holistic
    # pairing flic names with mediapipe pose indexes, from https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
    # flic_gt and mediapipe_landmarks are the keypoints for only one image

    if mediapipe_landmarks is None:
        # if the model does not find any pose, but we know there was ground truth
        return -math.inf

    gt_order = ["lsho", "lelb", "lwri", "rsho", "relb", "rwri", "lhip", "rhip", "leye", "reye", "nose"]
    gt_in_order = [flic_gt[joint] for joint in gt_order]
    flic_gt_array = np.stack([np.array([coord[0], coord[1]]) for coord in gt_in_order])
    mediapipe_order = [11, 13, 15, 12, 14, 16, 23, 24, 2, 5, 0]
    mediapipe_landmarks_array = np.stack([np.array([l[0], l[1]]) for l in mediapipe_landmarks[mediapipe_order]])

    if scoring_method == "pck3d":
        return pck3d(flic_gt_array, mediapipe_landmarks_array, normalization="flic_gt_torso")
    elif scoring_method == "2d_dist":
        return get_2d_landmark_diffs(flic_gt_array, mediapipe_landmarks_array, normalization="flic_gt_torso")
    else:
        raise ValueError(f"scoring method not recognized, got {scoring_method}")


# noinspection PyTypedDict
def mediapipe_to_dict(result):
    # We ignore the z values returned by holistic, since they are not accurate and we cannot easily visually evaluate it
    dict_res = {}
    if result.pose_landmarks:
        landmarks = np.stack([np.array([l.x, l.y, l.visibility]) for l in result.pose_landmarks.landmark])
        dict_res["pose"] = landmarks
    else:
        dict_res["pose"] = None

    if result.face_landmarks:
        landmarks = np.stack([np.array([l.x, l.y]) for l in result.face_landmarks.landmark])
        dict_res["face"] = landmarks
    else:
        dict_res["face"] = None

    if result.left_hand_landmarks:
        landmarks = np.stack(
            [np.array([l.x, l.y]) for l in result.left_hand_landmarks.landmark])
        dict_res["left_hand"] = landmarks
    else:
        dict_res["left_hand"] = None

    if result.right_hand_landmarks:
        landmarks = np.stack(
            [np.array([l.x, l.y]) for l in result.right_hand_landmarks.landmark])
        dict_res["right_hand"] = landmarks
    else:
        dict_res["right_hand"] = None
    return dict_res


def load_phoenix_images(data_type):
    """Returns all the images in the dataset in the format of a dict{video_folder_name: frames_files_names}"""

    phoenix_path = os.path.join("Datasets", "PHOENIX-2014-T-release-v3")
    data_dir = os.sep.join([phoenix_path, "PHOENIX-2014-T"])
    images_dir = os.sep.join([data_dir, "features", "fullFrame-210x260px", data_type])
    videos_names = os.listdir(images_dir)
    video_images_dict = OrderedDict()
    for video_path in tqdm(videos_names, desc=f"loading images dataset phoenix_{data_type}"):
        # Sort to ensure consistent sorting for glob.
        video_images_list = sorted(glob.glob(os.sep.join([images_dir, video_path, '*.png'])))
        video_images_dict[video_path] = [(img_path, None) for img_path in video_images_list]  # the None is because we have no ground truth
    return video_images_dict


def load_flic_images(data_type, dataset_name):
    """We return a dictionary of {videos: (frame_name, normalized_coords)}

    In the folder where the FLIC dataset was downloaded, the file demo_FLIC.m can be read for a matlab example
    Here we read the matlab struct with python
    We use the following fields of an annotation:
        moviename and filepath, to relate it to an img file
        poselet_hit_idx, coords, and torsobox can be used to compare mediapipe predictions with ground truth
        istrain, istest, isbad, isunchecked can be used to restrict to a subset of cleaner images

    The coords mentioned in the demo_FLIC.m are the ones labeled in mechanical turk, so those are the only ones we keep:
    lsho,  lelb,  lwri,  rsho,  relb,  rwri,  lhip,  rhip,  leye,  reye,  nose

    istrain - used for training/cross validation
    istest - used for testing
    isbad - not used, because at least one part is either mislabeled, occluded, very non-frontal or too tiny
    isunchecked - not used, because we didn't check whether it 'isbad' or not (simply ran out of time)
"""

    from scipy.io import loadmat
    flic_path = os.path.join("Datasets", "FLIC")
    full_or_normal_data_path = os.sep.join([flic_path, dataset_name])
    annotations_path = os.sep.join([full_or_normal_data_path, "examples.mat"])
    annotations = loadmat(annotations_path)
    annotations = annotations["examples"]  # it seems there is nothing useful besides the examples
    # to get filepath of frame(i), go to annotations[0][0]["filepath"]
    # the fields are: poselet_hit_idx, moviename, coords, filepath, imgdims, currframe, torsobox, istrain, istest, isbad, isunchecked

    images_dir = os.sep.join([full_or_normal_data_path, "images"])
    frames_names = os.listdir(images_dir)
    videos_names = set([re.search('(.+)-(\d+)\\.jpg', frame_name).group(1) for frame_name in frames_names])
    videos_names = sorted(list(videos_names))  # sort the videos so we always get them in the same order

    video_images_dict = OrderedDict()
    for video_name in tqdm(videos_names, desc=f"loading images dataset {dataset_name}_{data_type}"):

        # get only the annotations for the current video
        @np.vectorize
        def is_video(annotation): return annotation == video_name
        this_video_annotations = annotations[is_video(annotations["moviename"])]

        non_repeat_anots = np.array([anot for anot in this_video_annotations if
                              len(this_video_annotations[this_video_annotations["filepath"] == anot["filepath"]]) == 1])

        # we have the option of only using train or test images.
        if data_type == "train":
            this_type_anots = non_repeat_anots[np.array([elem.item() == 1 for elem in non_repeat_anots["istrain"]])]
        elif data_type == "test":
            this_type_anots = non_repeat_anots[np.array([elem.item() == 1 for elem in non_repeat_anots["istest"]])]
        else:
            this_type_anots = non_repeat_anots  # use them all

        # To understand how to read coords, read lookupPart.m, which has the indexes of the different parts of the body
        # coords is a ndarray of dims (2, 30). For the x and y coords of 30 points, but we only have 12 of those
        flic_joints_idxs = {"lsho": 1, "lelb": 2, "lwri": 3, "rsho": 4, "relb": 5, "rwri": 6, "lhip": 7,
                            "rhip": 10, "leye": 13, "reye": 14, "nose": 17}

        # we grab the annotated ground truth keypoints, being careful with 0 indexing
        imgs_coords = [(anot["filepath"], anot["imgdims"][0],
                        {key_joint: (anot["coords"][0, idx-1], anot["coords"][1, idx-1])  # grab the x and y coords
                         for key_joint, idx in flic_joints_idxs.items()}) for anot in this_type_anots]

        # the coords are (x, y) where x is how much we go down if we start at the left upper corner.
        # we normalize the coords using the size of the image to have them between 0 and 1. imgdims was (height, width)
        normalized_coords = [(os.sep.join([images_dir, info[0][0]]),
                             {key_joint: (coords[0] / info[1][1], coords[1] / info[1][0])
                             for key_joint, coords in info[2].items()})
                             for info in imgs_coords]

        video_images_dict[video_name] = normalized_coords

    return video_images_dict


def load_dataset_images(data_type, dataset_name):
    """Returns all the images in the dataset in the format of a
     dict{video_folder_name: (frames_files_names, extra_ground_truth_frame_info)}"""

    if dataset_name == "FLIC" or dataset_name == "FLIC-full":
        return load_flic_images(data_type, dataset_name)
    if dataset_name == "phoenix":
        return load_phoenix_images(data_type)
    else:
        raise "dataset not implemented, check spelling or add the dataset"


def create_or_load_results_file(results_pathname, results_columns):
    """Load previously computed results for the current run if present.

    Each run of an experiment using this test program will save the keypoints and diffs for each metamorphic rules
    in its own file. At the start of the run for the metrule, we load the dataset in the file, then for
    each frame we look if the keypoints for this frame and metRule (including no metRule) is already in the dataset or
    not. If it's already there we skip running it again, and we load the original keypoints to compute the diffs with
    other metRules. This allows to continue a run that was stopped in the middle, and it also allows to add new metRules
    easily since we can rerun the same script with all the rules and images without modification, and the program
    will simply skip the ones that are already computed."""

    results_path = pathlib.Path(results_pathname)
    if os.path.isfile(results_path):
        # load the dataset that was computed previously
        results = pd.read_pickle(results_path)
        # Early runs with phoenix dataset didn't have diff_with_gt column, but since it had no gt, we can just fill it
        # up with None. This can be removed if no old results files without None are being used.
        if 'diff_with_gt' not in results:
            results['diff_with_gt'] = None
    else:
        # create the results file with an empty dataframe for now with only the columns names
        results = pd.DataFrame(columns=results_columns)
        # Setting filenames and metRules as indexes will allow us to quickly check if each result was already computed
        results = results.set_index("input_name", drop=False)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_pickle(results_path)

    return results


def save_results_to_disk(new_results_list, results_columns, old_results_pd, results_path_of_current_rule):
    # TODO have to modify how we save now that we do csv instead of pickle, and saving all the separate diffs
    # TODO look at aux_scripts.convert_pickled_results_to_csv.py
    df_to_concat = pd.DataFrame(new_results_list, columns=results_columns)
    df_to_concat = df_to_concat.set_index("input_name", drop=False)
    if len(old_results_pd) != 0:
        combined_results = pd.concat([old_results_pd, df_to_concat])
    else:
        combined_results = df_to_concat
    # use temp file and os.replace to make the update atomic and never lose old results
    tempfile = results_path_of_current_rule + "temp"
    combined_results.to_pickle(tempfile)
    os.replace(tempfile, results_path_of_current_rule)


def process_video_no_metrule(video_images_list, holistic, dataset_name):
    rows_to_concat_result = []
    for sample_filepath, ground_truth_coords in video_images_list:
        # we used to check here if the frame sample_filepath was already computed, now this foo caller does it
        orig_img = cv2.imread(sample_filepath)
        # Convert the BGR image from opencv to RGB for mediapipe holistic before processing.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # If we didn't compute this example before, do it now
        orig_landmarks = holistic.process(orig_img)
        # noinspection PyTypeChecker
        orig_landmarks = mediapipe_to_dict(orig_landmarks)
        diff_with_ground_truth = diff_holistic_with_ground_truth(ground_truth_coords, orig_landmarks["pose"],
                                                                 dataset_name)
        rows_to_concat_result.append(["holistic", sample_filepath,
                                      "no-rule-orig-landmarks", None, diff_with_ground_truth, orig_landmarks])
    return rows_to_concat_result


def process_video_with_rule(video_images_list, holistic, rule_str, rule_instance, save_modified_images,
                            this_original_lndmrks, data_type, dataset_name, kwargs):
    rows_to_concat_result = []
    for sample_filepath, ground_truth_coords in video_images_list:
        # we used to check here if the frame sample_filepath was already computed, now this foo caller does it

        orig_img = cv2.imread(sample_filepath)
        # Convert the BGR image from opencv to RGB for mediapipe holistic before processing.
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Always modify the image instead of loading a previously modified version, we tested the times
        # and most filters are quicker or comparable to loading from disk
        rule_apply_kwargs = {"image_filepath": sample_filepath, "data_type": data_type} | kwargs
        modified_image = rule_instance.apply(orig_img, rule_apply_kwargs)

        if save_modified_images:
            # Currently we don't recommend saving the modified images during the main computation, they
            # can be generated in a temp folder during the visualization step if needed
            modified_image_path = aux_functions.get_modified_img_path(sample_filepath, rule_str, data_type, dataset_name)
            aux_functions.save_modified_image(modified_image, modified_image_path)

        modified_landmarks = holistic.process(modified_image)
        # noinspection PyTypeChecker
        modified_landmarks = mediapipe_to_dict(modified_landmarks)
        modified_landmarks = rule_instance.resize_keypoints(modified_landmarks, orig_img,
                                                            modified_image)
        diff_with_ground_truth = diff_holistic_with_ground_truth(ground_truth_coords, modified_landmarks["pose"],
                                                                 dataset_name)

        orig_landmarks = \
            this_original_lndmrks[this_original_lndmrks["input_name"] == sample_filepath]["system_outs"].iloc[0]
        diffs = {}
        for orig_landmark, modif_landmark, metric_type, diff_name in [(orig_landmarks["left_hand"],
                                                                       modified_landmarks["left_hand"],
                                                                       "hands", "left_hand"),
                                                                      (orig_landmarks["right_hand"],
                                                                       modified_landmarks["right_hand"],
                                                                       "hands", "right_hand"),
                                                                      (orig_landmarks["face"],
                                                                       modified_landmarks["face"],
                                                                       "face", "face"),
                                                                      (orig_landmarks["pose"],
                                                                       modified_landmarks["pose"],
                                                                       "pose", "pose")]:

            if orig_landmark is not None and modif_landmark is not None:
                diffs[diff_name] = holistic_metric(orig_landmark, modif_landmark, metric_type)
            elif orig_landmark is None and modif_landmark is None:
                diffs[diff_name] = math.nan
            elif orig_landmark is None and modif_landmark is not None:
                # if the model identifies this feature in the modified image but not in the original img
                diffs[diff_name] = math.inf
            elif orig_landmark is not None and modif_landmark is None:
                # if the model identifies this feature in the original image but not in the modified img
                diffs[diff_name] = -math.inf
            else:
                raise "the if else logic of the landmarks is wrong, we should never reach this line"

        rows_to_concat_result.append(["holistic", sample_filepath, rule_str, diffs,
                                      diff_with_ground_truth, modified_landmarks])
    return rows_to_concat_result


def process_video_gral(arguments):
    # pool.imap apparently only gives one argument to the function to be called
    (frames_to_process, video_path, this_original_landmarks, rule_str, save_modified_images,
     doing_orig_image, data_type, dataset_name, kwargs) = arguments

    with (mp.solutions.holistic.Holistic(
            static_image_mode=True,
            # False for videos, True for unrelated static imgs. Video has nondeterminism probl
            model_complexity=2) as holistic):
        if doing_orig_image:
            list_of_video_results = process_video_no_metrule(frames_to_process, holistic, dataset_name)
        else:
            rule_instance = instantiate_rules.met_rule_result_str_to_instance(rule_str)

            list_of_video_results = process_video_with_rule(frames_to_process, holistic, rule_str,
                                                            rule_instance, save_modified_images,
                                                            this_original_landmarks, data_type, dataset_name, kwargs)
    return list_of_video_results


def run_multi_process_all_videos(phoenix_dict, doing_orig_image, rule_str, save_modified_images, old_results_pd,
                                 how_many_videos_to_process, results_columns, results_path_of_current_rule,
                                 all_original_landmarks, small_sample, data_type, kwargs, cpu_nums, dataset_name):

    videos_processed = 0
    got_new_results = False
    this_run_results_list = []

    # check which videos have already been totally computed, for those, don't even create a process since it takes time
    # For the others, save which frames we have to process
    all_videos = set(phoenix_dict.keys())
    frames_left_to_process = []
    for video in all_videos:
        this_video_frames = phoenix_dict[video]  # this_video_frames has (frame_filepath, ground_truth_coords), coords can be None
        total_frames_num = len(this_video_frames)

        # Count how many of the current video frames were already present in the old results
        frames_already_computed = old_results_pd.index.isin([name for name, _ in this_video_frames])
        num_frames_already_computed = frames_already_computed.sum()
        already_computed_frames_names = old_results_pd.index[frames_already_computed]

        # also save the original landmarks corresponding to this video, so we don't copy the whole thing for the
        # subprocesses. If we are computing the original landmarks now, then pass a None value
        this_original_lndmrks = (None if all_original_landmarks is None else
                                 all_original_landmarks[all_original_landmarks['input_name'].str.contains(video,
                                                                                                         na=False)])

        if num_frames_already_computed == 0:
            # if no frames were processed before
            frames_left_to_process.append((video, this_original_lndmrks, this_video_frames))
        elif num_frames_already_computed < total_frames_num:
            # if some frames are still missing, add these to the list of frames to process
            frames_left_to_process.append((video, this_original_lndmrks,
                                           [frame for frame in this_video_frames if frame not in already_computed_frames_names]))

    print(f"from a total of {len(all_videos)} videos, we still have {len(frames_left_to_process)} to process")

    try:
        print(f"maximum cpu_count: {multiprocessing.cpu_count()}; current max multiprocessing: {cpu_nums}")
        with multiprocessing.Pool(cpu_nums, maxtasksperchild=5) as pool:
            for result in tqdm(pool.imap_unordered(func=process_video_gral,
                                                   iterable=[(frames_list, video_path,
                                                              original_lndmrks, rule_str, save_modified_images,
                                                              doing_orig_image, data_type, dataset_name, kwargs)
                                                             for (video_path, original_lndmrks, frames_list)
                                                             in frames_left_to_process]),
                               total=len(frames_left_to_process),
                               desc=f"processing videos in {dataset_name} for metRule {rule_str}"):
                # when each of the processes finishes and we get the result for each video, we aggregate all results in the
                # manager class
                videos_processed += 1
                got_new_results = got_new_results or len(result) != 0  # check if we got new results
                this_run_results_list += result
                del result

                if got_new_results and videos_processed % 250 == 0:
                    # once in many videos, save the results
                    save_results_to_disk(this_run_results_list, results_columns, old_results_pd,
                                         results_path_of_current_rule)

                if small_sample and videos_processed > how_many_videos_to_process:
                    # in case we wanted just a small sample, we kill all remaining processes when we have what we need
                    break
    except Exception as e:
        print("Found error while processing videos, terminating pool")
        pool.terminate()
        raise e

    # Save final results after all the video_path for the current met_rule
    if got_new_results:
        save_results_to_disk(this_run_results_list, results_columns, old_results_pd,
                             results_path_of_current_rule)
        print(f"saved new results to {results_path_of_current_rule}")

    return this_run_results_list


def test_holistic(met_rules, results_filepath, kwargs):
    """Test mediapipe holistic model against PHOENIX dataset metamorphic image modifications

    :param met_rules: list of metamorphic rules to use to modify the images
    :param kwargs:
        :small_sample: for quick debugging, if True only process some videos
        :data_type: Use the videos from dev, test or train folder
        :save_modified_images: save the modified images to disk for later viewing
    """

    small_sample = kwargs.get('small_sample', False)
    data_type = kwargs.get('data_type', 'test')
    dataset_name = kwargs.get('dataset_name', 'phoenix')
    save_modified_images = kwargs.get('save_modified_images', False)
    cpu_nums = kwargs.get('multiprocess_count', math.floor(multiprocessing.cpu_count() * 3 / 4))

    results_columns = ["system", "input_name", "met_rules", "eval_diffs", "diff_with_gt", "system_outs"]

    dataset_imgs_dict = load_dataset_images(data_type, dataset_name)

    how_many_videos_to_process = 2 if small_sample else 2 * len(dataset_imgs_dict)

    rule_str = instantiate_rules.rule_and_kwargs_to_complex_rule_str(met_rules)

    # results will be saved in a folder like this example: {out_dir}/run1-phoenix-dev/run1-phoenix-dev-{rule}
    run_description = results_filepath.split(os.sep)[-1] + f"-{dataset_name}-{data_type}"
    results_folder = results_filepath.removesuffix(results_filepath.split(os.sep)[-1]) + run_description + "/"
    results_path_of_current_rule = results_folder + run_description + "-" + rule_str
    old_results = create_or_load_results_file(results_path_of_current_rule, results_columns)

    no_rule_str = "no-rule-orig-landmarks"
    doing_orig_image = rule_str == no_rule_str
    # save all the original_landmarks to be compared against when doing other metRules
    all_original_landmarks = pd.read_pickle(results_path_of_current_rule.removesuffix(rule_str) + no_rule_str)\
                             if not doing_orig_image else None

    new_results = run_multi_process_all_videos(dataset_imgs_dict, doing_orig_image, rule_str, save_modified_images,
                                               old_results, how_many_videos_to_process, results_columns,
                                               results_path_of_current_rule, all_original_landmarks, small_sample,
                                               data_type, kwargs, cpu_nums, dataset_name)

    if len(new_results) != 0:
        df_to_concat = pd.DataFrame(new_results, columns=results_columns)
        df_to_concat = df_to_concat.set_index("input_name", drop=False)
        if len(old_results) != 0:
            return pd.concat([old_results, df_to_concat])
        else:
            return df_to_concat
    else:
        return old_results
