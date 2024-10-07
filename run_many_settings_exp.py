#!/usr/bin/env python3
import argparse
import datetime
import traceback

import os
import sys

from main_test import main_test

settings_per_rule = {
    # first we always need to run the 'no-rule-orig-landmarks' to get the original results
    "no-rule-orig-landmarks": [[]],

    # runs without masks
    "identity": [[]],
    "img-stretch": [["height_ratio: 1.05", "width_ratio: 0.95"],
                    ["height_ratio: 0.95", "width_ratio: 1.05"],
                    ["height_ratio: 1.1", "width_ratio: 0.9"],
                    ["height_ratio: 0.9", "width_ratio: 1.1"],
                    ["height_ratio: 1.25", "width_ratio: 1"],
                    ["height_ratio: 1", "width_ratio: 1.25"],
                    ["height_ratio: 1.4", "width_ratio: 1"],
                    ["height_ratio: 1", "width_ratio: 1.4"],
                    ["height_ratio: 0.8", "width_ratio: 1"],
                    ["height_ratio: 1", "width_ratio: 0.8"],
                    ["height_ratio: 0.6", "width_ratio: 1"],
                    ["height_ratio: 1", "width_ratio: 0.6"]
                    ],
    "img-rotation": [['rotation_angle: 5', 'center: (0.5, 0.5)'],
                     ['rotation_angle: 10', 'center: (0.5, 0.5)'],
                     ['rotation_angle: 15', 'center: (0.5, 0.5)'],
                     ['rotation_angle: 25', 'center: (0.5, 0.5)'],
                     ],
    "img-blur": [['blur_strength: 80', 'filter_size: 5'],
                 ['blur_strength: 80', 'filter_size: 3'],
                 ['blur_strength: 80', 'filter_size: 7'],
                 ['blur_strength: 80', 'filter_size: 9'],
                 ['blur_strength: 10', 'filter_size: 5'],
                 ['blur_strength: 10', 'filter_size: 3'],
                 ['blur_strength: 10', 'filter_size: 7'],
                 ['blur_strength: 10', 'filter_size: 9'],
                 ['blur_strength: 50', 'filter_size: 5'],
                 ['blur_strength: 50', 'filter_size: 3'],
                 ['blur_strength: 50', 'filter_size: 7'],
                 ['blur_strength: 50', 'filter_size: 9'],
                 ['blur_strength: 30', 'filter_size: 5'],
                 ['blur_strength: 30', 'filter_size: 3'],
                 ['blur_strength: 30', 'filter_size: 7'],
                 ['blur_strength: 30', 'filter_size: 9'],
                 ['blur_strength: 150', 'filter_size: 5'],
                 ['blur_strength: 150', 'filter_size: 3'],
                 ['blur_strength: 150', 'filter_size: 7'],
                 ['blur_strength: 150', 'filter_size: 9'],
                 ['blur_strength: 180', 'filter_size: 5'],
                 ['blur_strength: 180', 'filter_size: 3'],
                 ['blur_strength: 180', 'filter_size: 7'],
                 ['blur_strength: 180', 'filter_size: 9'],
                 ['blur_strength: 125', 'filter_size: 5'],
                 ['blur_strength: 125', 'filter_size: 3'],
                 ['blur_strength: 125', 'filter_size: 7'],
                 ['blur_strength: 125', 'filter_size: 9']],
    "img-motion-blur": [['kernel_size: 7', 'orientation: 0'],
                        ['kernel_size: 5', 'orientation: 0'],
                        ['kernel_size: 9', 'orientation: 0'],
                        ['kernel_size: 11', 'orientation: 0'],
                        ['kernel_size: 7', 'orientation: 70'],
                        ['kernel_size: 5', 'orientation: 70'],
                        ['kernel_size: 9', 'orientation: 70'],
                        ['kernel_size: 11', 'orientation: 70'],
                        ['kernel_size: 7', 'orientation: 100'],
                        ['kernel_size: 5', 'orientation: 100'],
                        ['kernel_size: 9', 'orientation: 100'],
                        ['kernel_size: 11', 'orientation: 100'],
                        ['kernel_size: 7', 'orientation: 40'],
                        ['kernel_size: 5', 'orientation: 40'],
                        ['kernel_size: 9', 'orientation: 40'],
                        ['kernel_size: 1', 'orientation: 40']],
    "img-mirror": [["flip_code: 1"],
                   ["flip_code: 0"],
                   ["flip_code: -1"]],
    "img-black-white": [[]],
    "img-resolution": [['size_ratio: 0.98'],
                       ['size_ratio: 0.95'],
                       ['size_ratio: 0.9'],
                       ['size_ratio: 0.8'],
                       ['size_ratio: 0.7'],
                       ['size_ratio: 0.6'],
                       ['size_ratio: 0.5'],
                       ['size_ratio: 0.4'],
                       ['size_ratio: 0.3'],
                       ['size_ratio: 0.2'],
                       ['size_ratio: 0.1']],
    "img-dark-bright": [['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 1.15', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 0.85', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 1.5', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 0.5', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 1.05', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 0.95', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 1.75', 'method: gamma'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 0.25', 'method: gamma'],
                        ['brightness_constant: 0', "brightness_multiplier: 1.15", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.6", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 20', "brightness_multiplier: 0.8", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 0', "brightness_multiplier: 1.05", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 20', "brightness_multiplier: 1.2", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 20', "brightness_multiplier: 0.4", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: 30', "brightness_multiplier: 1.15", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: -20', "brightness_multiplier: 1.6", 'gamma: 1.2', 'method: scale'],
                        ['brightness_constant: -20', "brightness_multiplier: 0.8", 'gamma: 1.2', 'method: scale'],
                        ],

    # runs with masks
    # 'mask_type from 'skin', 'clothes', 'hair', 'background'
    "img-masked": [  # silhouette runs
                   ["mask_type: clothes", "applied_rule: img-silhouette", 'color_constant: (0, 0, 255)'],
                   ["mask_type: skin", "applied_rule: img-silhouette", 'color_constant: (0, 0, 255)'],
                   ["mask_type: background", "applied_rule: img-silhouette", 'color_constant: (0, 0, 255)'],
                   ["mask_type: clothes", "applied_rule: img-silhouette", 'color_constant: (255, 180, 120)'],  # color close to skin, to confuse on purpose
                   ["mask_type: background", "applied_rule: img-silhouette", 'color_constant: (255, 180, 120)'],  # color close to skin, to confuse on purpose
                   ["mask_type: skin", "applied_rule: img-silhouette", 'color_constant: (255, 180, 120)'],  # if we put same color clothes, but no shadows, does it affect it?
                   ["mask_type: skin", "applied_rule: img-silhouette", 'color_constant: (230, 120, 120)'],
                   ["mask_type: background", "applied_rule: img-silhouette", 'color_constant: (33, 28, 27)'],  # color close to clothes, to confuse on purpose
                   ["mask_type: skin", "applied_rule: img-silhouette", 'color_constant: (33, 28, 27)'],  # color close to clothes, to confuse on purpose
                   ["mask_type: clothes", "applied_rule: img-silhouette", 'color_constant: (33, 28, 27)'],  # if we put same color clothes, but no shadows, does it affect it?

                   # color wheel runs
                   ["mask_type: skin", "applied_rule: img-color-wheel", 'hue_rotation: 90'],
                   ["mask_type: clothes", "applied_rule: img-color-wheel", 'hue_rotation: 90'],
                   ["mask_type: background", "applied_rule: img-color-wheel", 'hue_rotation: 90'],
                   ["mask_type: hair", "applied_rule: img-color-wheel", 'hue_rotation: 90'],
                   ["mask_type: skin", "applied_rule: img-color-wheel", 'hue_rotation: 30'],
                   ["mask_type: clothes", "applied_rule: img-color-wheel", 'hue_rotation: 30'],
                   ["mask_type: skin", "applied_rule: img-color-wheel", 'hue_rotation: 10'],
                   ["mask_type: clothes", "applied_rule: img-color-wheel", 'hue_rotation: 10'],
                   ["mask_type: skin", "applied_rule: img-color-wheel", 'hue_rotation: -45'],
                   ["mask_type: clothes", "applied_rule: img-color-wheel", 'hue_rotation: -45'],

                   # color channel runs
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (0.45, 1, 1.2)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (1.2, 1, 0.45)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (1.4, 1, 0.6)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (0.6, 1.4, 1)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (1.3, 1.3, 0.8)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (0.8, 1.3, 1.3)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (1.1, 1.1, 0.9)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: RGB', 'channel_multipliers: (0.9, 1.1, 1.1)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: XYZ', 'channel_multipliers: (1, 1, 1)'],
                   ["mask_type: skin", "applied_rule: img-color-channels", 'color_channels_out: BGR', 'channel_multipliers: (1, 1, 1)'],
                ],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-results_folder', '--results_folder', type=str,  required=True,
                        help='string path to the folder where the results and the logs will be saved. The results'
                             'might occupy 50kb per image in inputs')
    parser.add_argument('-no_phoenix', '--no_phoenix', action="store_true",
                        help='if this flag is used, the phoenix dataset will be skipped during processing')
    args = parser.parse_args()

    results_folder = args.results_folder
    current_time = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
    log_file = f"{results_folder}/logfiles/{current_time}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print(f"saving detailed logs in {log_file}")

    skip_phoenix = args.no_phoenix
    if skip_phoenix:
        datasets = [("FLIC", "test")]
    else:
        datasets = [("FLIC", "test"), ("phoenix", "dev")]

    for dataset, data_type in datasets:
        for rule, settings in settings_per_rule.items():
            for single_run_settings in settings:
                out_file_prefix = f"{results_folder}/"
                # it doesn't always save to the same file, to the out_file the rule and settings are appended inside the script
                config_file_path = "./runs_settings/exp2_current_config.yml"

                command = rule
                for setting in single_run_settings:
                    command += " " + setting
                currently_running_message = f"=======running {command} for {dataset} {data_type}========"
                print(currently_running_message)

                with open(config_file_path, "w") as config_file:
                    config_file.write("---\n")
                    config_file.write(f"out_dir: {out_file_prefix}\n")
                    config_file.write(f"sys_test: holistic\n")
                    config_file.write(f"data_type: {data_type}\n")
                    config_file.write(f"dataset_name: {dataset}\n")
                    config_file.write(f"met_rules:\n")
                    config_file.write(f"    - {rule}:\n")
                    for setting in single_run_settings:
                        config_file.write(f"        {setting}\n")

                # save outputs into a log file so we can later check if something failed what it was
                with open(log_file, "a") as f:
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = f
                    sys.stderr = f
                    try:
                        print(currently_running_message)
                        results = main_test(config_file_path)
                    except Exception as e:
                        print(''.join(traceback.TracebackException.from_exception(e).format()))
                        sys.stderr.flush()
                        print("Found an error, finishing the execution")
                        sys.stdout.flush()
                        exit()

                    sys.stderr.flush()
                    print("finished at least one run")
                    sys.stdout.flush()

                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

                print("Current setting done")
            print("outside loop")
        print(f"Finished all settings for {dataset}")

    print("Finished all the settings in the current experiment")