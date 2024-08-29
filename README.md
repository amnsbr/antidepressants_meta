# Meta-analytic Functional Effects of Antidepressants in Depression

This repository contains the code and data associated with the paper "Convergent Functional Effects of Antidepressants in Major Depressive Disorder: A Neuroimaging Meta-analysis" by Saberi et al.

## Repository Structure
- `figures`: Contains Jupyter Notebooks used to generate the figures for the paper.
- `input`: Contains subdirectories for each ALE analysis. Each subdirectory includes two Excel files formatted according to the [pyALE requirements](https://github.com/LenFrahm/pyALE/blob/main/ALE.ipynb):
  - `coordinates.xlsx`: Lists the coordinates along with their associated experiments, contrasts, and number of subjects.
  - `analysis.xlsx`: Defines the analyses to be performed.
- `scripts`:
  - `setup`: Includes `setup.sh`, which creates a virtual environment, installs dependencies from `requirements.txt`, and clones `pyALE` into the `tools` directory.
  - `ale`: Contains `run.py` for executing all ALE meta-analyses using `pyALE`, along with `run.submit` and a wrapper bash script `run.sh` for running the meta-analyses on a compute cluster via HTCondor.
  - `macm`: Contains scripts for running MACM analysis on the left DLPFC cluster identified in the ALE performed on the Treated > Untreated contrast.
  - `ccm`: Includes `ccm.py` which runs Convergent Connectivity Mapping using either random-effects (`ccm_random`) or fixed-effects (`ccm_fixed`) analysis. This script is intended to be used as an imported module rather than a standalone script.
  - `helpers`: A module containing helper functions that can be imported by other scripts:
    - `io.py`: Functions for loading data used in the analyses.
    - `plot.py`: Functions for plotting cortical surfaces and subcortical structures (as meshes or volumes).
    - `stats.py`: Statistical functions for spatial associations between different maps, accounting for spatial autocorrelation.
    - `transform.py`: Functions for transforming data between different representations (HCP Cifti, fsLR, MNI, and parcellated formats).
  - `src`: Includes additional data files required for running the analyses, transformations, and plotting.
  - `download_hcp_dense.py`: Downloads the HCP dense connectome (29 GB) to the specified directory. Note that this file is not included in the repository due to its large size and restricted access. You must have an account at https://db.humanconnectome.org/ to download it. Usage: `python download_hcp_dense.py`.
- `tools`: Contains the specific commit of [`pyALE`](https://github.com/LenFrahm/pyALE/tree/ff7cdf5b50e242a6e20aadabf22dc820732fa5fa) used in this project.

## Additional Requirements
In addition to the Python dependencies specified in `requirements.txt` and the `tools/pyALE` directory, the following are required for the scripts to run:

- The HCP dense connectome must be downloaded and available at the path specified by the environment variable `$HCP_DENSE_PATH`. Use `scripts/download_hcp_dense.py` to download this data.
- The BrainMap dataset is expected to be located at `scripts/src/BrainMap_dump_Feb2024_mask-Grey10.pkl.gz`. Due to access restrictions, this file is not shared in the repository but can be provided upon request. Note that the BrainMap database is searchable using [Sleuth](https://www.brainmap.org/sleuth/).
- `figures/set_env.py` must exist and define the environment variables `$PROJECT_DIR` (the project's root directory) and `$HCP_DENSE_PATH` (the path to the HCP dense connectome file). This script is not included in the repository as it contains specific paths used in our institute's cluster.

## Support
If you have any questions, feel free to contact me at amnsbr\[at\]gmail.com, a.saberi\[at\]fz-juelich.de, or saberi\[at\]cbs.mpg.de.
