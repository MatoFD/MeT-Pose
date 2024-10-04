The code to reproduce results reported in "Metamorphic Testing for Pose Estimation Systems" for ICST2025 \
The project works on Linux systems

===== To install the conda environment ==== \
conda env create -f=./MeT-Pose-env.yml \
conda activate MeT-Pose

===== Datasets set up ====
- For the PHOENIX-T dataset (around 45gb), it can be downloaded from https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
    and should be set up in the folder structure as follows: ./Datasets/PHOENIX-2014-T-release-v3
- For the FLIC and FLIC-full dataset, it can be downloaded from https://bensapp.github.io/flic-dataset.html
  Download the file FLIC.zip (around 300mb) and paste in the folder structure as follows: ./Datasets/FLIC/

The Dataset structure before running the scripts should look like:

<pre>
project-folder
├─ other-scripts
└─ Datasets
    ├─ FLIC
    │   └─ FLIC 
    │       ├─ images 
    │       ├─ demo_flic.m 
    │       ├─ examples.mat 
    │       ├─ lookupPart.m 
    │       ├─ myplot.m 
    │       └─ plotbox.m 
    └─ PHOENIX-2014-T-release-v3 
        ├─ README 
        └─ PHOENIX-2014-T 
            ├─ annotations 
            ├─ evaluation 
            ├─ features 
            └─ models 
</pre>

===== Running the main files to generate the landmarks and differences ====

Running "python get_paper_plots.py" will run through the experiments and save the csv outputs used for the plots in the paper.
Might need more than 100GB for phoenix-dev results. Alternatively, can also run "python get_paper_plots.py -no_phoenix" for only the FLIC results which run faster and need less space.
If executed with argument "-save_imgs" it saves examples of modified images.

The intermediate csvs and dataframes will be saved in ./Results/, and the final csvs used for the reported plots, and similar plots will be saved in ./plots/
Some examples of modified images will be saved in ./modified_images/

===== details, not needed to compute the report results =====

The arguments that need to be specified in the YAML configuration file can be seen in main_test.py in the main function as argparse arguments, so they show up when doing main_test.py -h.

For now the main processing in the project is all done inside test_mediapipe_holistic.py. If the project is expanded beyond holistic this would no longer be the case.

The main scripts to call when conducting experiments are main_test.py, to get the actual landmarks from mediapipe holistic.
Although we call it using run_many_settings_exp.py, which sets up a yaml configuration and calls main_test with many settings.

Note that for all of this, and for visualising the results,
first the script should be run with "no-rule-orig-landmarks" as the metamorphic rule, to get
the landmarks for the images without modification. The project could be modified to compute this if it is called with
a metamorphic rule before the original ones are calculated, but it hasn't been done so far.

