import subprocess

rules_per_group = {
    "AllRels": "AllRels",
    "SubRels": ["identity", "img-black-white", "no-rule-orig-landmarks",
               "img-blur_10_3", "img-blur_125_5", "img-blur_80_7",
               "img-dark-bright_20_0.8_1.2_scale", "img-dark-bright_20_1.2_0.5_gamma",
               "img-mirror_1",
               "img-motion-blur_11_0", "img-motion-blur_11_100", "img-resolution_0.2",
               "img-resolution_0.7", "img-rotation_5_(0.5, 0.5)", "img-rotation_10_(0.5, 0.5)",
               "img-stretch_1.25_1", "img-stretch_1_0.6", "img-stretch_1_0.8",
               "img-masked_background_img-color-wheel_90", "img-masked_hair_img-color-wheel_90", ],
    "MirrorH": ["img-mirror_1"],
    "Greyscale": ["img-black-white"],
    "GreyAndMirr": ["img-mirror_1", "img-black-white"],
    "AllMotionBlur": ["all_rules_in_img-motion-blur"]
}


#possible_features = ["left_hand", "right_hand", "face", "pose", "gt"]
#possible_image_aggregations = ["mean", "median", "max", "min"]
#possible_cross_img_aggregations = ["table_subsumption_ratios", "histogram_failed_images_across_thresholds",
#                                   "gt_vs_metamorphic_failed_tests_across_thresholds", "images_failed_per_num_of_rules"]

#change if needed
#res_dir = './Results/'
#data_dir = './Datasets/'
#img_aggr = 'median'
#save_mod_imgs = True

print("doing RQ1 results")
for rules_codename in ['AllRels', 'SubRels', "GreyAndMirr"]:
    for dataset in ['FLIC-test', 'phoenix-dev']:
        subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose',
                        '-plot_type=histogram_failed_images_across_thresholds', f'-rules_codename={rules_codename}',
                        f'metR={rules_per_group[rules_codename]}'])

for rules_codename in ["MirrorH", "Greyscale"]:
    dataset = 'phoenix-dev'
    subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=l_hand', 'r_hand',
                    '-plot_type=histogram_failed_images_across_thresholds', f'-rules_codename={rules_codename}',
                    f'metR={rules_per_group[rules_codename]}'])

print("doing RQ2 results")
for rules_codename in ['AllRels', 'SubRels', "Greyscale", "MirrorH"]:
    dataset = 'FLIC-test'
    subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose', 'gt',
                    '-plot_type=gt_vs_metamorphic_failed_tests_across_thresholds', f'-rules_codename={rules_codename}',
                    f'metR={rules_per_group[rules_codename]}'])

print("doing RQ3 results")
for rules_codename in ['SubRels', "AllMotionBlur"]:
    for dataset in ['FLIC-test', 'phoenix-dev']:
        subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose'
                        '-plot_type=table_subsumption_ratios', f'-rules_codename={rules_codename}',
                        f'metR={rules_per_group[rules_codename]}'])

rules_codename = "AllRels"
for dataset in ['FLIC-test', 'phoenix-dev']:
    subprocess.run(['python', './make_plots.py', f'-dataset={dataset}', '-kp_type=pose'
                    '-plot_type=images_failed_per_num_of_rules', f'-rules_codename={rules_codename}',
                    f'metR={rules_per_group[rules_codename]}'])
