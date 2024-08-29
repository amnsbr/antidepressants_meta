import sys
import os
import types
import numpy as np
import pandas as pd
import scipy
import nibabel
from matplotlib.colors import LinearSegmentedColormap

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PYALE_DIR = os.path.join(PROJECT_DIR, "tools", "pyALE")
INPUT_DIR = os.path.join(PROJECT_DIR, "input")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
SRC_DIR = os.path.join(PROJECT_DIR, "scripts", "src")
HCP_DIR = os.environ.get("HCP_DIR")

sys.path.append(os.path.join(PYALE_DIR, "utils"))
from tal2icbm_spm import tal2icbm_spm

from . import transform


def load_coordinates(
    exp_subdir, contrast, by_experiment=False, return_experiment_info=False
):
    """
    Loads coordinates from a given ALE experiment

    Parameters
    ----------
    exp_subdir: str
        Name of the experiment subdirectory in the ALE input folder
    contrast: str
        Name of the contrast, if '' all contrasts are loaded
    by_experiment: bool
        If True, returns a groupby object with experiments as keys
    return_experiment_info: bool
        If True, returns the experiment info DataFrame

    Returns
    -------
    coordinates: np.ndarray
        Shape (n_coords, 3)
    or
    experiment_info: pd.DataFrame
    or
    experiment_info_groupby: pd.DataFrameGroupBy
    """
    path = os.path.join(INPUT_DIR, exp_subdir)
    experiment_info_path = os.path.join(path, "coordinates.xlsx")
    experiment_info = pd.read_excel(os.path.join(path, experiment_info_path))
    experiment_info = experiment_info.dropna(subset=["Experiments"])
    ## convert TAL to MNI
    if (experiment_info["Space"] == "TAL").sum() > 0:
        experiment_info.loc[experiment_info["Space"] == "TAL", ["X", "Y", "Z"]] = (
            tal2icbm_spm(
                experiment_info.loc[
                    experiment_info["Space"] == "TAL", ["X", "Y", "Z"]
                ].values
            )
        )
    experiment_info["Space"] = "MNI"
    ## get the contrast if indicated
    if contrast:
        experiment_info = experiment_info.loc[experiment_info["Contrast"] == contrast]
    if return_experiment_info:
        return experiment_info
    else:
        if by_experiment:
            return experiment_info.groupby("Experiments")
        else:
            ## get the coordiantes
            return experiment_info.loc[:, ["X", "Y", "Z"]].values


def load_yeo_map():
    """
    Loads 7 resting state networks map of Yeo et al. 2011

    Returns
    -------
    yeo_map: a name space with the following attributes
        - cifti: cifti map
        - categorical: map as categorical pandas Series
        - names
        - shortnames
        - colors
        - cmap
    """
    # load cifti
    yeo_cifti = nibabel.load(
        os.path.join(SRC_DIR, "Yeo2011_7Networks_N1000.dlabel.nii")
    )
    # convert to categorical series
    yeo_categorical = pd.Series(yeo_cifti.get_fdata().flatten().astype("int")).astype(
        "category"
    )
    yeo_names = [
        "None",
        "Visual",
        "Somatomotor",
        "Dorsal attention",
        "Ventral attention",
        "Limbic",
        "Frontoparietal",
        "Default",
    ]
    yeo_shortnames = ["NA", "VIS", "SMN", "DAN", "SAN", "LIM", "FPN", "DMN"]
    yeo_categorical = yeo_categorical.cat.rename_categories(yeo_names)
    yeo_categorical[yeo_categorical == "None"] = np.NaN
    yeo_categorical = yeo_categorical.cat.remove_unused_categories()
    # creat colormap
    yeo_colors = np.array(
        [
            l.rgba
            for i, l in list(
                yeo_cifti.header.get_index_map(0)._maps[0].label_table.items()
            )[1:]
        ]
    )
    yeo_colors = yeo_colors[:, :3]
    yeo_cmap = LinearSegmentedColormap.from_list("yeo", yeo_colors, 7)
    return types.SimpleNamespace(
        cifti=yeo_cifti,
        categorical=yeo_categorical,
        names=yeo_names,
        shortnames=yeo_shortnames,
        colors=yeo_colors,
        cmap=yeo_cmap,
    )


def load_pet_maps(schaefer="400", tian="S2"):
    """
    Loads parcellated PET maps

    Parameters
    ----------
    schaefer: {'400'}
    tian: {'S1', 'S2'}

    Returns
    -------
    pet_data: (pd.DataFrame)
        Shape (n_parc, n_receptors)
    """
    out_path = os.path.join(SRC_DIR, f"PET_parc-sch{schaefer}_tian{tian}.csv")
    if os.path.exists(out_path):
        return pd.read_csv(out_path, index_col=0)
    parcellated_data = pd.DataFrame()
    # get pet metadata
    metadata = pd.read_csv(
        os.path.join(SRC_DIR, "PET_nifti_images_metadata.csv"), index_col="filename"
    )
    # group the images with the same recetpro-tracer
    for group, group_df in metadata.groupby(["receptor"]):
        group_name = group
        # take a weighted average of PET value z-scores
        # across images with the same receptor-tracer
        # (weighted by N of subjects)
        pet_parcellated_weighted_sum = {}
        for filename, file_metadata in group_df.iterrows():
            print(filename)
            filepath = os.path.join(SRC_DIR, "PET_nifti_images", filename)
            pet_parcellated = transform.parcellate_volumetric(
                filepath, schaefer=schaefer, tian=tian, nonzero=True
            )
            pet_parcellated_weighted_sum[filename] = (
                scipy.stats.zscore(pet_parcellated.values, nan_policy="omit")
                * file_metadata["N"]
            )[:, np.newaxis]
        # divide the sum of weighted Z-scores by total N
        parcellated_data.loc[:, group_name] = (
            np.nansum(
                np.concatenate(list(pet_parcellated_weighted_sum.values()), axis=1),
                axis=1,
            )
            / group_df["N"].sum()
        )
    # Z-score the merged pet maps across parcels (in the
    # case of one map per receptor this will not do anything
    # since the maps are already Z-scored above)
    parcellated_data = scipy.stats.zscore(parcellated_data)
    # add labels of the parcels
    parcellated_data.index = pet_parcellated.index
    parcellated_data.to_csv(out_path, index_label="parcel")
    return parcellated_data
