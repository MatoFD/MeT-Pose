The main use for this project is run by calling
python main_test.py -config=config_file_path

===== To install the conda environment ====
conda install --file=./test-ignorance-assumption-env.yml


===== Running the main files to generate the landmarks and differences ====

The arguments that need to be specified in the YAML configuration file can be seen in main_test.py in the main function
as argparse arguments, so they show up when doing main_test.py -h.

For now the main processing in the project is all done inside test_mediapipe_holistic.py. If the project is expanded
beyond holistic this would no longer be the case.

The main scripts to call when conducting experiments are main_test.py, to get the actual landmarks from mediapipe holistic.
Although we call it using run_many_settings_exp.py, which sets up a yaml configuration and calls main_test with many settings.

Note that for all of this, and for visualising the results,
first the script should be run with "no-rule-orig-landmarks" as the metamorphic rule, to get
the landmarks for the images without modification. The project could be modified to compute this if it is called with
a metamorphic rule before the original ones are calculated, but it hasn't been done so far.

