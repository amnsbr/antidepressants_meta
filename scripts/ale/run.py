import os
import sys

PYALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "pyALE")
sys.path.append(PYALE_DIR)
from nb_pipeline import setup, analysis

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "input")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "ale")

# get the list of analyses (one subfolder per each) from input directory
analyses = sorted(os.listdir(INPUT_DIR))              
for analysis_name in analyses:
    path = os.path.join(RESULTS_DIR, analysis_name)
    if os.path.exists(os.path.join(path, 'Results')):
        print(
            f"{analysis_name} is already done or started"
            f"\n If you wish to repeat it remove {os.path.join(path, 'Results')}"
        )
        continue
    # load data
    analysis_info_filename = os.path.join(INPUT_DIR, analysis_name, 'analysis.xlsx')
    experiment_info_filename = os.path.join(INPUT_DIR, analysis_name, 'coordinates.xlsx')
    print(path, analysis_info_filename, experiment_info_filename)
    meta_df, exp_all, tasks = setup(path, analysis_info_filename, experiment_info_filename)

    # run ALE
    analysis(path = path,
             meta_df = meta_df,
             exp_all = exp_all,
             tasks = tasks,
             tfce_enabled = True,
             null_repeats = 10000,
             cluster_thresh = 0.001,
             sample_n = 2500,
             nprocesses = 6)