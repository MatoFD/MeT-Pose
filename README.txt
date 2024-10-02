The main use for this project is run by calling
python main_test.py -config=config_file_path

===== To install the conda environment ====
conda env create -f=./MeT-Pose-env.yml
conda activate MeT-Pose

===== Running the main files to generate the landmarks and differences ====

Running "python get_paper_plots.py" will run through the experiments and save the csv outputs used for the plots in the paper.
Might need up to 100GB for phoenix-dev results. Alternativelly, can also run "python get_paper_plots.py -no_phoenix" for only the FLIC results which run faster and need less space.


===== details, not needed to compute the reporte results =====

The arguments that need to be specified in the YAML configuration file can be seen in main_test.py in the main function as argparse arguments, so they show up when doing main_test.py -h.

For now the main processing in the project is all done inside test_mediapipe_holistic.py. If the project is expanded beyond holistic this would no longer be the case.

The main scripts to call when conducting experiments are main_test.py, to get the actual landmarks from mediapipe holistic.
Although we call it using run_many_settings_exp.py, which sets up a yaml configuration and calls main_test with many settings.

Note that for all of this, and for visualising the results,
first the script should be run with "no-rule-orig-landmarks" as the metamorphic rule, to get
the landmarks for the images without modification. The project could be modified to compute this if it is called with
a metamorphic rule before the original ones are calculated, but it hasn't been done so far.

