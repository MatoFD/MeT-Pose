#!/usr/bin/env python3

import argparse
import inspect
import math
import multiprocessing

import yaml

from test_systems.test_system import test_system
from metamorphic_rules.instantiate_rules import metamorphic_rules_dict


def main_test(config_file):
    with open(config_file) as f:
        yaml_dict = yaml.safe_load(f)

    kwargs = yaml_dict
    out_dir = kwargs["out_dir"]
    met_rules = kwargs["met_rules"]
    sys_test = kwargs["sys_test"]

    return test_system(met_rules, sys_test, out_dir, kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The inputs of the script will be read from a yaml file, the '
                                                 'descriptions of the individual arguments should be used in the yaml'
                                                 'The only argument that will be read here is config, all the others'
                                                 'are just for documentation purposes.'
                                                 'Also, by using main_test.py [met_rule] -h, you can see a description'
                                                 'of all the arguments for that met_rule')

    parser.add_argument('-config', '--config-file', type=str,  # required=True,
                        help='Path to the yaml file detailing all the required arguments for the current run.'
                             'Check the readme for details on how it can be configured')
    parser.add_argument('-in', '--in_dir', type=str,
                        help='Path to the folder containing the images to test. '
                             'Can be ignored depending on the dataset')
    parser.add_argument('-out', '--out_dir', type=str,  # required=True,
                        help='Required output filepath for saving the resulting differences and keypoints')
    parser.add_argument('--met_rules', type=str, nargs="+", choices=metamorphic_rules_dict.keys(),  # required=True,
                        help='List of metamorphic rules to use for this test. When using more than one rule, they '
                             'should all be right after "met_rules" and before all the specific kwargs for each rule')
    parser.add_argument('--sys_test', type=str, choices=["holistic"],  # required=True,
                        help='ML system to test. Current available systems are: holistic')
    parser.add_argument('-small', '--small_sample', action="store_true",
                        help='If true, will only use a few images from the dataset, helps with quick testing. '
                             'Default False')
    parser.add_argument('-data', '--data_type', type=str,
                        help='specify which folder inside the dataset to use:'
                             ' (train, test, dev) for phoenix; or (train, test, all) for FLIC/FLIC-full')
    parser.add_argument('-dataset', '--dataset_name', type=str,
                        help='specify which dataset we are using (phoenix, FLIC, FLIC-full, etc.)')
    parser.add_argument('-save_imgs', '--save_modified_images', action="store_true",
                        help='If true, save to disk all of the modified versions of the original images after the '
                             'chosen metamorphic rule, this enables later visualization but takes up the same space on'
                             ' disk as the original dataset (does not save images again if they are already saved).'
                             'Default False for batch runs, set to True for manual verification')
    parser.add_argument('-cores', '--multiprocess_count', action="store_true",
                        default=math.floor(multiprocessing.cpu_count() * 3/4),
                        help='Select the number of child processes used when paralelising the execution of the script,'
                             'recommended and default values are half of your available cores:'
                             f'{math.floor(multiprocessing.cpu_count() * 3/4)}')

    # help messages for each metRule. Originally seemed like the way to take the kwargs for each metRule
    subparsers = parser.add_subparsers(help='We have subparsers for detailing the kwargs expected by each of the '
                                            'different metamorphic rules')
    rules_subparsers = []
    for rule_name, rule_class in metamorphic_rules_dict.items():
        new_subparser = subparsers.add_parser(rule_name, help=inspect.getdoc(rule_class))
        kwargs_info = rule_class.kwargs_constructor_list()
        for kwarg_name, kwarg_type, kwarg_help, kwarg_default in kwargs_info:
            new_subparser.add_argument(f"-{kwarg_name}", f"--{kwarg_name}", type=kwarg_type, help=kwarg_help,
                                       default=kwarg_default)
        rules_subparsers.append(new_subparser)

    # Check which metRules are in the program arguments and parse their relevant kwargs
    recognized_args, unrecognized_args = parser.parse_known_args()

    main_test(recognized_args.config_file)

