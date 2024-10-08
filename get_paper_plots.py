import subprocess
import sys

SubRels = ("identity,img-black-white,img-blur_10_3,img-blur_125_5,img-blur_80_7,"
           "img-dark-bright_20_0.8_1.2_scale,img-dark-bright_20_1.2_0.5_gamma,"
           "img-masked_background_img-color-wheel_90,img-masked_hair_img-color-wheel_90,"
           "img-mirror_1,"
           "img-motion-blur_11_0,img-motion-blur_11_100,img-resolution_0.2,"
           "img-resolution_0.7,'img-rotation_5_(0.5, 0.5)','img-rotation_10_(0.5, 0.5)',"
           "img-stretch_1.25_1,img-stretch_1_0.6,img-stretch_1_0.8"
           )
rules_per_group = {
    "AllRels": "AllRels",
    "SubRels": SubRels,
    "SubRelsGT": SubRels+",no-rule-orig-landmarks",
    "MirrorH": "img-mirror_1",
    "MirrorHGT": "img-mirror_1,no-rule-orig-landmarks",
    "Greyscale": "img-black-white",
    "GreyscaleGT": "img-black-white,no-rule-orig-landmarks",
    "GreyAndMirr": "img-mirror_1,img-black-white",
    "AllMotionBlur": "all_rules_in_img-motion-blur",
    "FailedPerNumRulesTable": "FailedPerNumRulesTable"
}

# by default, do all the experiment including phoenix-dev, unless the user wants to skip it due to time and space constraints
if "-no_phoenix" in sys.argv:
    include_phoenix_processing = False
    print("skipping phoenix, only doing FLIC")
else:
    include_phoenix_processing = True

# by default, we do not need to save modified images every time
if "-save_imgs" in sys.argv:
    save_mod_imgs = True
    print("Saving modified images")
else:
    save_mod_imgs = False


#============ First run the testing framework on the SUT to get the raw results information to analyse
subprocess_commands = ['python', 'run_many_settings_exp.py', '-results_folder=./Results/']
if not include_phoenix_processing:
    subprocess_commands += ["-no_phoenix"]
subprocess.run(subprocess_commands)


#=========== Second call convert_pickled_results_to_csv to prepare the raw_diffs.csv files that we need to aggregate the data
# as reported in this work.
print("translating flic dataframes to csvs")
subprocess.run(['python', 'convert_pickled_results_to_csv.py', '-res_dir=./Results/', '-dataset=FLIC', '-data_type=test'])

if include_phoenix_processing:
    print("translating phoenix dataframes to csvs")
    subprocess.run(['python', 'convert_pickled_results_to_csv.py', '-res_dir=./Results/', '-dataset=phoenix', '-data_type=dev'])


#============ Lastly get the specific results reported in this work

#possible_features = ["left_hand", "right_hand", "face", "pose", "gt"]
#possible_image_aggregations = ["mean", "median", "max", "min"]
#possible_cross_img_aggregations = ["table_subsumption_ratios", "histogram_failed_images_across_thresholds",
#                                   "gt_vs_metamorphic_failed_tests_across_thresholds", "images_failed_per_num_of_rules"]

#change if needed
#res_dir = './Results/'
#data_dir = './Datasets/'
#img_aggr = 'median'
#save_mod_imgs = False

print("doing RQ1 results")
for rules_codename in ['AllRels', 'SubRels', "GreyAndMirr"]:
    if include_phoenix_processing:
        datasets = ['FLIC-test', 'phoenix-dev']
    else:
        datasets = ['FLIC-test']
    for dataset in datasets:
        if save_mod_imgs:
            subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose', '-save_mod_imgs',
                            '-plot_type=histogram_failed_images_across_thresholds', f'-rules_codename={rules_codename}',
                            f'-metR={rules_per_group[rules_codename]}'])
        else:
            subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose',
                            '-plot_type=histogram_failed_images_across_thresholds',
                            f'-rules_codename={rules_codename}',
                            f'-metR={rules_per_group[rules_codename]}'])

if include_phoenix_processing:
    for rules_codename in ["MirrorH", "Greyscale"]:
        dataset = 'phoenix-dev'
        subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=left_hand', '-kp_type=right_hand',
                        '-plot_type=histogram_failed_images_across_thresholds',
                        f'-rules_codename={rules_codename}',
                        f'-metR={rules_per_group[rules_codename]}'])

print("doing RQ2 results")
for rules_codename in ['AllRels', 'SubRelsGT', "GreyscaleGT", "MirrorHGT"]:
    dataset = 'FLIC-test'
    subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose', '-kp_type=gt',
                    '-plot_type=gt_vs_metamorphic_failed_tests_across_thresholds',
                    f'-rules_codename={rules_codename}',
                    f'-metR={rules_per_group[rules_codename]}'])

print("doing RQ3 results")
for rules_codename in ['SubRels', "AllMotionBlur"]:
    if include_phoenix_processing:
        datasets = ['FLIC-test', 'phoenix-dev']
    else:
        datasets = ['FLIC-test']
    for dataset in datasets:
        subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose',
                        '-plot_type=table_subsumption_ratios',
                        f'-rules_codename={rules_codename}',
                        f'-metR={rules_per_group[rules_codename]}'])

rules_codename = "FailedPerNumRulesTable"
for dataset in datasets:
    subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose',
                    '-plot_type=images_failed_per_num_of_rules',
                    f'-rules_codename={rules_codename}',
                    f'-metR={rules_per_group[rules_codename]}'])
