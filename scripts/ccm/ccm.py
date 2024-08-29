import sys
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel
import nimare.transforms
import statsmodels.stats.multitest
from tqdm import tqdm

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

sys.path.append(os.path.join(PROJECT_DIR, "scripts"))
from helpers import io, transform


def ccm_random(
    exp_subdir=None,
    contrast="",
    experiment_info=None,
    dconn_path=os.environ.get("HCP_DENSE_PATH"),
    distance_threshold=10,
    n_perm=1000,
    weighted=True,
    z_from_p=True,
    seed=0,
    save_null=True,
    override=False,
):
    """
    Convergent connectivity mapping of given coordinates based
    on HCP dense connectome and following an adaptation of the approach
    introduced by Cash et al. 2023 (https://doi.org/10.1038/s44220-023-00038-8).
    Here the combined RSFC is calculated first across the foci of each
    experiment separately and then combined across experiments (random-effects analysis)

    Parameters
    -------
    exp_subdir: (str)
        it must exist under io.RESULTS_DIR/ale
    contrast: (str | None)
    experiment_info: (None | pd.DataFrame)
        if provided exp_subdir and contrast will be ignored
    dconn_path: (str)
        Path to HCP dense connectome file
    distance_threshold: (float)
        if distance of a focus and its assigned grayordinate in mm
        is higher than this threshold it will be excluded
    n_perm: (int)
    weighted: (bool)
        weight experiments by their sample sizes
    z_from_p: (bool)
        calculate z from non-parametric p-values
        rather than as `(obs - mean(null)) / std(null)`
    seed: (int)
    save_null: (bool)
        saves null distribution to disk
    override: (bool)

    Returns
    -------
    zmap: (np.ndarray)
        convergent connectivity z map
    """
    # specify output path
    subdir = "ccm_random"
    if weighted:
        subdir += "_weighted"
    else:
        subdir += "_nonweighted"
    subdir += f"_n-{n_perm}"
    out_path = os.path.join(io.RESULTS_DIR, "ccm", subdir, exp_subdir, contrast)
    if os.path.exists(os.path.join(out_path, "zmap.npy")) and not override:
        print("Already done!")
        return np.load(os.path.join(out_path, "zmap.npy"))
    print("Results will be saved in", out_path)
    if experiment_info is None:
        # load true coordiantes
        experiment_info = io.load_coordinates(
            exp_subdir, contrast, return_experiment_info=True
        )
    # load dense connectome
    dconn = nibabel.load(dconn_path)
    # load MNI coordinates of grayordinates
    all_xyz = transform.get_cifti_mni_coordinates()
    # step 1: calculate true (observed) convergent
    # connectivity map separately within each experiment
    # and take their (weighted) average
    true_points = {}
    true_points_dist = {}
    n_points = {}
    n_subs = {}
    mean_fcs = {}
    sum_mean_fcs = np.zeros(dconn.dataobj.shape[0])
    sum_mean_fcs_denom = 0
    print("Calculating true FC")
    for experiment, experiment_df in tqdm(experiment_info.groupby("Experiments")):
        # map MNI coordinates to closest grayordinates (points)
        coordinates = experiment_df[["X", "Y", "Z"]].values
        dist_mat = scipy.spatial.distance_matrix(coordinates, all_xyz)
        true_points[experiment] = dist_mat.argmin(axis=1)
        closest_dist = dist_mat.min(axis=1)
        true_points[experiment] = true_points[experiment][
            closest_dist < distance_threshold
        ]
        true_points_dist[experiment] = closest_dist[closest_dist < distance_threshold]
        n_points[experiment] = true_points[experiment].size
        n_subs[experiment] = experiment_df.iloc[0]["Subjects"]
        # calculate average dense RSFC of current
        # experiment foci
        sum_fc = np.zeros(dconn.dataobj.shape[0])
        for point in true_points[experiment]:
            sum_fc += dconn.dataobj[:, point]
            sum_fc[point] -= dconn.dataobj[point, point]  # subtract self-connection
        mean_fcs[experiment] = sum_fc / n_points[experiment]
        # add up this experiment's convergent connectivity
        if weighted:
            sum_mean_fcs += n_subs[experiment] * mean_fcs[experiment]
            sum_mean_fcs_denom += n_subs[experiment]
        else:
            sum_mean_fcs += mean_fcs[experiment]
            sum_mean_fcs_denom += 1
    # calculate the (weighted) mean of convergent
    # connectivity maps across included experiments
    mean_mean_fcs = sum_mean_fcs / sum_mean_fcs_denom
    print(f"Points: {sum(n_points.values())} of {experiment_info.shape[0]}")
    distance_stats = pd.Series(
        np.concatenate(list(true_points_dist.values()))
    ).describe()
    print("Closest distance of included points (in mm)\n", distance_stats)
    # save observed outputs
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "true_fc-mean.npy"), mean_mean_fcs)
    pd.Series(n_points).to_csv(os.path.join(out_path, "n_points.csv"))
    distance_stats.to_csv(os.path.join(out_path, "distance_stats.csv"))
    for experiment in mean_fcs.keys():
        np.save(
            os.path.join(out_path, f"true_fc-{experiment}.npy"), mean_fcs[experiment]
        )
    # step 2: similarly calculate null convergent connectivity
    # stratified by experiemtns
    np.random.seed(seed)
    mean_fc_null = np.zeros((n_perm, dconn.dataobj.shape[0])) * np.NaN
    for perm_idx in tqdm(range(n_perm)):
        null_mean_fcs = {}
        null_sum_mean_fcs = np.zeros(dconn.dataobj.shape[0])
        # create null (simulated) set of coordinates with the same
        # number of grayordinates per experiment
        for experiment in n_points.keys():
            rand_points = np.random.choice(
                np.arange(dconn.dataobj.shape[0]), size=n_points[experiment]
            )
            sum_fc = np.zeros(dconn.dataobj.shape[0])
            for point in rand_points:
                sum_fc += dconn.dataobj[:, point]
                sum_fc[point] -= dconn.dataobj[point, point]  # subtract self-connection
            null_mean_fcs[experiment] = sum_fc / n_points[experiment]
            if weighted:
                null_sum_mean_fcs += n_subs[experiment] * null_mean_fcs[experiment]
            else:
                null_sum_mean_fcs += null_mean_fcs[experiment]
        # note that there is no need to recalculate denominator as it is
        # identical to observed
        mean_fc_null[perm_idx] = null_sum_mean_fcs / sum_mean_fcs_denom
    if save_null:
        np.save(os.path.join(out_path, "null_fcs.npy"), mean_fc_null)
    if z_from_p:
        zmap_name = "zmap"
        # calculate asymmetric one-tailed p-values
        p_right = ((mean_fc_null >= mean_mean_fcs).sum(axis=0) + 1) / (n_perm + 1)
        p_left = ((-mean_fc_null >= -mean_mean_fcs).sum(axis=0) + 1) / (n_perm + 1)
        # calculate two-tailed p-values
        p = np.minimum(p_right, p_left) * 2
        # convert z to p
        zmap = nimare.transforms.p_to_z(p, tail="two")
        # make z of voxels with more extreme values
        # towards left tail negative
        zmap[p_left < p_right] *= -1
    else:
        zmap_name = "zmap_from_std"
        diff_map = mean_mean_fcs - mean_fc_null.mean(axis=0)
        zmap = diff_map / mean_fc_null.std(axis=0)
    np.save(os.path.join(out_path, f"{zmap_name}.npy"), zmap)
    return zmap


def ccm_fixed(
    exp_subdir=None,
    contrast="",
    coordinates=None,
    dconn_path=os.environ.get("HCP_DENSE_PATH"),
    distance_threshold=10,
    n_perm=1000,
    z_from_p=True,
    seed=0,
    save_null=True,
    override=False,
):
    """
    Convergent connectivity mapping of given coordinates based
    on HCP dense connectome and following an adaptation of the approach
    introduced by Cash et al. 2023 (https://doi.org/10.1038/s44220-023-00038-8).
    Here the average RSFC is calculated across all the foci regardless
    of their experiments as fixed effects

    Parameters
    -------
    exp_subdir: (str)
        it must exist under io.RESULTS_DIR/ale
    contrast: (str | None)
    coordinates: (None | pd.DataFrame)
        if provided exp_subdir and contrast will be ignored
    dconn_path: (str)
        Path to HCP dense connectome file
    distance_threshold: (float)
        if distance of a focus and its assigned grayordinate in mm
        is higher than this threshold it will be excluded
    n_perm: (int)
    z_from_p: (bool)
        calculate z from non-parametric p-values
        rather than as `(obs - mean(null)) / std(null)`
    seed: (int)
    save_null: (bool)
        saves null distribution to disk
    override: (bool)

    Returns
    -------
    zmap: (np.ndarray)
        convergent connectivity z map
    """
    out_path = os.path.join(
        io.RESULTS_DIR, "ccm", f"ccm_fixed_n-{n_perm}", exp_subdir, contrast
    )
    if os.path.exists(os.path.join(out_path, "zmap.npy")) and not override:
        print("Already done!")
        return np.load(os.path.join(out_path, "zmap.npy"))
    print("Results will be saved in", out_path)
    if coordinates is None:
        # load true coordiantes
        coordinates = io.load_coordinates(exp_subdir, contrast)
    # load coordiantes of CIFTI coordinates and
    # get closest CIFTI grayordinates to each coordinate
    all_xyz = transform.get_cifti_mni_coordinates()
    dist_mat = scipy.spatial.distance_matrix(coordinates, all_xyz)
    true_points = dist_mat.argmin(axis=1)
    closest_dist = dist_mat.min(axis=1)
    true_points = true_points[closest_dist < distance_threshold]
    n_points = true_points.size
    print(f"Points: {n_points} of {coordinates.shape[0]}")
    distance_stats = pd.Series(
        closest_dist[closest_dist < distance_threshold]
    ).describe()
    print("Closest distance of included points (in mm)\n", distance_stats)
    # load dense connectome
    dconn = nibabel.load(dconn_path)
    # calculate true FC
    print("Calculating true FC")
    sum_fc = np.zeros(dconn.dataobj.shape[0])
    for point in true_points:
        sum_fc += dconn.dataobj[:, point]
        sum_fc[point] -= dconn.dataobj[point, point]  # subtract self-connection
    mean_fc = sum_fc / n_points
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "true_fc.npy"), mean_fc)
    pd.Series(n_points).to_csv(os.path.join(out_path, "n_points.csv"))
    distance_stats.to_csv(os.path.join(out_path, "distance_stats.csv"))
    # calculate null FCs
    np.random.seed(seed)
    mean_fc_null = np.zeros((n_perm, dconn.dataobj.shape[0])) * np.NaN
    for perm_idx in tqdm(range(n_perm)):
        rand_points = np.random.choice(np.arange(dconn.dataobj.shape[0]), size=n_points)
        sum_fc_null = np.zeros(dconn.dataobj.shape[0])
        for point in rand_points:
            sum_fc_null += dconn.dataobj[:, point]
            sum_fc_null[point] -= dconn.dataobj[
                point, point
            ]  # subtract self-connection
        mean_fc_null[perm_idx] = sum_fc_null / n_points
    if save_null:
        np.save(os.path.join(out_path, "null_fcs.npy"), mean_fc_null)
    if z_from_p:
        zmap_name = "zmap"
        # calculate asymmetric one-tailed p-values
        p_right = ((mean_fc_null >= mean_fc).sum(axis=0) + 1) / (n_perm + 1)
        p_left = ((-mean_fc_null >= -mean_fc).sum(axis=0) + 1) / (n_perm + 1)
        # calculate two-tailed p-values
        p = np.minimum(p_right, p_left) * 2
        # convert z to p
        z_unsigned = nimare.transforms.p_to_z(p, tail="two")
        # specify negative z voxels (with more extreme values
        # towards left tail)
        neg = p_left < p_right
        zmap = z_unsigned.copy()
        zmap[neg] *= -1
    else:
        zmap_name = "zmap_from_std"
        diff_map = mean_fc - mean_fc_null.mean(axis=0)
        zmap = diff_map / mean_fc_null.std(axis=0)
    np.save(os.path.join(out_path, f"{zmap_name}.npy"), zmap)
    return zmap


def ccm_yeo(true_fc, null_fcs, fdr=True, plot=True, ax=None):
    """
    Convergent connectivity mapping resolved across
    resting state networks of Yeo et al. 2011
    based on true and null FCs calculated using
    `ccm_fixed` and `ccm_random`

    Parameters
    ----------
    true_fc: (np.ndarray)
        true (observed) convergent FC in Cifti space.
        Shape (91282,)
    null_fcs: (np.ndarray)
        null FCs in Cifti space.
        Shape (n_perm, 91282)
    fdr: (bool)
        Apply FDR correction on p values
    plot: (bool)
    ax: (None | AxesSubplot)

    Returns
    -------
    p_vals: (pd.Series)
    true_fc_yeo: (np.ndarray)
        true (observed) convergent FC averaged across Yeo networks.
        Shape (7,)
    null_fcs_yeo:
        null FCs averaged across Yeo networks.
        Shape (n_perm, 7)
    """
    # transform to fsLR
    true_fc_fslr = transform.cifti_to_fsLR(true_fc)
    null_fcs_fslr = transform.cifti_to_fsLR(null_fcs)
    # load yeo map
    yeo_map = io.load_yeo_map()
    # average across yeo networks
    true_fc_yeo = (
        pd.DataFrame(true_fc_fslr, index=yeo_map.categorical)
        .reset_index()
        .groupby("index")
        .mean()
    )
    null_fcs_yeo = (
        pd.DataFrame(null_fcs_fslr.T, index=yeo_map.categorical)
        .reset_index()
        .groupby("index")
        .mean()
    )
    # calculate p-values per network
    n_perm = null_fcs.shape[0]
    p_vals = (
        (np.abs(null_fcs_yeo.values) > np.abs(true_fc_yeo.values)).sum(axis=1) + 1
    ) / (n_perm + 1)
    if fdr:
        _, p_vals = statsmodels.stats.multitest.fdrcorrection(p_vals)
    p_vals = pd.Series(p_vals, index=yeo_map.names[1:])
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        sns.violinplot(
            data=null_fcs_yeo.unstack().reset_index(),
            x="index",
            y=0,
            split=True,
            gap=0,
            alpha=1.0,
            linewidth=0.5,
            inner="point",
            inner_kws=dict(s=1, alpha=0.5),
            ax=ax,
        )
        # Note: if colors are set within violinplot
        # the half-violins will face each other
        # this is a workaround: paint them after plotting
        for i, violin in enumerate(ax.collections[::2]):
            for j in range(len(violin.get_paths())):
                violin.set_facecolor(yeo_map.colors[i])
        plt.setp(ax.collections, zorder=0, label="")  # puts violins in the back
        # plot observed
        ax.scatter(
            x=np.arange(7) + 0.45, y=true_fc_yeo[0], marker=1, color="black", s=40
        )
        # aesthetics
        sns.despine(ax=ax, offset=10, trim=True)
        ax.set_ylabel("Mean RSFC (Z)", fontsize=18)
        ax.set_xlabel("")
        xticklabels = yeo_map.shortnames[1:]
        for i in range(7):
            if p_vals[i] < 0.05:
                xticklabels[i] += "\n*"
        ax.set_xticklabels(xticklabels, fontsize=18)
    return p_vals, true_fc_yeo, null_fcs_yeo
