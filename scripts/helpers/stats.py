import os
import sys
import numpy as np
import pandas as pd
import scipy
import nibabel
import nilearn.surface
import neuromaps.nulls
import brainsmash
from tqdm import tqdm

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC_DIR = os.path.join(PROJECT_DIR, "scripts", "src")

from . import transform, io


def spin_surf_data(surf_data, n_perm=1000):
    """
    Spins fsLR surface data using pre-calculated spins of indices

    Parameters
    ----------
    surf_data: (np.ndarray)
        must be in fsLR 32k space; Shape: (64984,)
    n_perm: (int)

    Returns
    -------
    surf_data_spin: (np.ndarray)
        Shape: (64984, n_perm)
    """
    # create or load spins of indices
    if not os.path.exists(os.path.join(SRC_DIR, f"fsLR_spin_n-{n_perm}.npz")):
        idx_orig = np.tile(np.arange(transform.N_VERTICES_HEM_FSLR), 2)
        idx_spin = neuromaps.nulls.alexander_bloch(
            idx_orig, "fslr", "32k", seed=0, n_perm=n_perm
        )
        np.savez_compressed(
            os.path.join(SRC_DIR, f"fsLR_spin_n-{n_perm}.npz"), idx_spin=idx_spin
        )
    else:
        idx_spin = np.load(os.path.join(SRC_DIR, f"fsLR_spin_n-{n_perm}.npz"))[
            "idx_spin"
        ]
    # spin data
    surf_data_spin = surf_data[idx_spin]
    return surf_data_spin

def get_parcel_centers(schaefer="400", tian="S2", method="mean"):
    """
    Finds center of cortex-subcortex parcels in 2mm MNI space

    Parameters
    ---------
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    method: {'mean', 'min_dist'}
        - 'mean': mean XYZ of all parcel voxels
        - 'min_dist': finds the voxel with lowest
            sum of distance to all other voxels

    Returns
    -------
    centers: (pd.DataFrame)
        xyz coordinates of parcel centers
        Shape: (n_parc, 3)
    """
    if tian is None:
        parc_filename_mni = (
            f"Schaefer2018_{schaefer}Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
        )
        parc_filename_cifti = (
            f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"
        )
    else:
        parc_filename_mni = f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}_MNI152NLin6Asym_2mm.nii.gz"
        parc_filename_cifti = f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii"
    parc_path = os.path.join(io.SRC_DIR, parc_filename_mni)
    out_path = parc_path.replace(".nii.gz", f"_centers_{method}.csv")
    # load it if already exists
    if os.path.exists(out_path):
        return pd.read_csv(out_path, index_col="parcel")
    # get labels excluding background
    atlas_cifti = nibabel.load(os.path.join(io.SRC_DIR, parc_filename_cifti))
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ][1:]
    parc_img = nibabel.load(parc_path)
    parc_img_data = parc_img.get_fdata()
    centers = pd.DataFrame(index=labels, columns=["x", "y", "z"], dtype=float)
    for i, label in enumerate(labels):
        parc = i + 1
        if method == "mean":
            voxel_ind_center = np.vstack(np.where(parc_img_data == parc)).mean(axis=1)
        elif method == "min_dist":
            parc_voxels = np.vstack(np.where(parc_img_data == parc)).T
            distances = scipy.spatial.distance.pdist(
                np.squeeze(parc_voxels), "euclidean"
            )  # Returns condensed matrix of distances
            distancesSq = scipy.spatial.distance.squareform(
                distances
            )  # convert to square form
            sumDist = np.sum(distancesSq, axis=1)  # sum distance across columns
            index = np.where(sumDist == np.min(sumDist))  # minimum sum distance index
            voxel_ind_center = parc_voxels[index].squeeze()
        centers.loc[label] = nibabel.affines.apply_affine(
            parc_img.affine, voxel_ind_center
        )
    centers.to_csv(out_path, index_label="parcel")
    return centers


def get_hem_parcels(schaefer="400", tian="S2"):
    """
    Returns lists of L and R parcels

    Parameters
    ---------
    schaefer: {'400'}
    tian: {'S1', 'S2'}

    Returns
    -------
    hem_parcels: (dict)
        with 'L' and 'R' keys including
        an array of parcel labels in each hemisphere
    """
    if tian is None:
        parc_filename = f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"
    else:
        parc_filename = f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii"
    atlas_cifti = nibabel.load(os.path.join(io.SRC_DIR, parc_filename))
    # get labels excluding midline '???'
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ][1:]
    labels = np.array(labels)
    hem_parcels = {}
    for hem in ["L", "R"]:
        hem_parcels[hem] = labels[
            np.where(pd.Series(labels).str.lower().str.contains(f"{hem.lower()}h"))[0]
        ]
    return hem_parcels


def calculate_euclidean_dist(schaefer="400", tian="S2", center_method="mean"):
    """
    Calculate Euclidean distance between all pairs
    of parcels, separately within each hemisphere

    Parameters
    ---------
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    center_method: {'mean', 'min_dist'}
        method used to identify parcel centers
        - 'mean': mean XYZ of all parcel voxels
        - 'min_dist': finds the voxel with lowest
            sum of distance to all other voxels

    Returns
    -------
    dist_mats: (dict)
        with 'L' and 'R' keys including
        distance matrices of each hemisphere
    """
    # get parcel center coordinates
    centers = get_parcel_centers(schaefer, tian, method=center_method)
    # specifiy which parcels belong to L and R
    hem_parcels = get_hem_parcels(schaefer, tian)
    # calculate euclidean distance of parcel
    # centers
    dist_mats = {}
    for hem in ["L", "R"]:
        hem_centers = centers.loc[hem_parcels[hem]].values
        dist_mat = scipy.spatial.distance_matrix(hem_centers, hem_centers)
        dist_mats[hem] = pd.DataFrame(
            dist_mat, index=hem_parcels[hem], columns=hem_parcels[hem]
        )
    return dist_mats


def get_variogram_surrogates(
    X,
    schaefer="400",
    tian="S2",
    n_perm=1000,
    center_method="mean",
    surrogates_path=None,
):
    """
    Creates or loads SA-preserved surrogates of parcellated data
    using variograms

    Parameters
    ----------
    X: (pd.DataFrame)
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    n_perm: (int)
    center_method: {'mean', 'min_dist'}
        method used to identify parcel centers
        - 'mean': mean XYZ of all parcel voxels
        - 'min_dist': finds the voxel with lowest
            sum of distance to all other voxels
    surrogates_path: (None | str)

    Returns
    -------
    surrogates: (np.ndarray)
        Shape (n_perm, n_parc)
    parcels: (np.ndarray)
        Shape (n_parc,)
    """
    create_surrogates = True
    if (surrogates_path is not None) and (os.path.exists(surrogates_path)):
        surrogates = np.load(surrogates_path, allow_pickle=True)["surrogates"]
        parcels = np.load(surrogates_path, allow_pickle=True)["parcels"]
        print(f"Surrogates already exist in {surrogates_path}")
        create_surrogates = False
    if create_surrogates:
        print(f"Creating {n_perm} surrogates based on variograms in {surrogates_path}")
        # load distance matrices
        dist_mats = calculate_euclidean_dist(
            schaefer, tian, center_method=center_method
        )
        # get parcels per hemisphere and split the data
        hem_parcels = get_hem_parcels(schaefer, tian)
        X_hems = {
            "L": X.loc[hem_parcels["L"], :].values,
            "R": X.loc[hem_parcels["R"], :].values,
        }
        surrogates = {}
        for hem in ["L", "R"]:
            # load geodesic distance matrices for each hemisphere
            dist_mat = dist_mats[hem].values
            # initialize the surrogates
            surrogates[hem] = np.zeros(
                (n_perm, X_hems[hem].shape[0], X_hems[hem].shape[1])
            )
            for col_idx in range(X_hems[hem].shape[1]):
                # create surrogates per each column
                base = brainsmash.mapgen.base.Base(
                    x=X_hems[hem][:, col_idx], D=dist_mat, seed=0
                )
                surrogates[hem][:, :, col_idx] = base(n=n_perm)
        # concatenate hemispheres
        surrogates = np.concatenate(
            [surrogates[hem] for hem in ["L", "R"]], axis=1
        )  # axis 1 is the parcels
        parcels = np.concatenate([hem_parcels["L"], hem_parcels["R"]])
        if surrogates_path:
            np.savez_compressed(surrogates_path, surrogates=surrogates, parcels=parcels)
    return surrogates, parcels


def variogram_test(
    X,
    Y,
    schaefer="400",
    tian="S2",
    method="pearson",
    n_perm=1000,
    center_method="mean",
    surrogates_path=None,
):
    """
    Calculates non-parametric p-value of correlation between the columns in X and Y
    by creating surrogates of X with their spatial autocorrelation preserved based
    on variograms. Note that X and Y must be parcellated.

    Parameters
    ----------
    X: (pd.DataFrame)
    Y: (pd.DataFrame)
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    method: {'pearson', 'kendall', 'spearman'} or callable
        see `pandas.DataFrame.corr`
    n_perm: (int)
    center_method: {'mean', 'min_dist'}
        method used to identify parcel centers
        - 'mean': mean XYZ of all parcel voxels
        - 'min_dist': finds the voxel with lowest
            sum of distance to all other voxels
    surrogates_path: (None | str)

    Returns
    -------
    test_r, p_vals: (pd.DataFrame)
        Shape (n_cols_X, n_cols_Y)
    null_distribution: (np.ndarray)
        Shape (n_perm, n_cols_Y, n_cols_X)
    """
    # step 0: reorder X and Y to L->R
    # following the order of surrogates
    X = X.copy()
    Y = Y.copy()
    X["orig_order"] = Y["orig_order"] = np.arange(X.shape[0])
    X["rh"] = Y["rh"] = X.index.str.lower().str.contains("rh")
    X = X.sort_values(by=["rh", "orig_order"]).drop(columns=["rh", "orig_order"])
    Y = Y.sort_values(by=["rh", "orig_order"]).drop(columns=["rh", "orig_order"])
    assert np.all(X.index == Y.index)
    # step 1: create or load surrogates
    surrogates, parcels = get_variogram_surrogates(
        X,
        schaefer,
        tian,
        n_perm,
        center_method=center_method,
        surrogates_path=surrogates_path,
    )
    assert np.all(parcels == X.index)
    # step 2: calculate test correlation coefficient between all pairs of columns between X and Y
    coefs = (
        pd.concat([X, Y], axis=1)
        # calculate the correlation coefficient between all pairs of columns within and between X and Y
        .corr(method=method)
        # select only the correlations we are interested in
        .iloc[: X.shape[1], -Y.shape[1] :]
        # convert it to shape (1, n_features_Y, n_features_surface_X)
        .T.values[np.newaxis, :]
    )
    # step 3: permutate across surrogates and create null distribution
    null_distribution = np.zeros((n_perm, Y.shape[1], X.shape[1]))
    for x_col in range(X.shape[1]):
        # get all surrogate parcellated maps at once.
        # this involves complicated indexing but the is best way
        # to achieve this extremely more efficiently than loops.
        # if the first permutation in surrogates is e.g. [335, 212, ...]
        # it basically assigns the X 0th value to 335th parcel and
        # its 1st to 212th parcel and so on, and does the same across
        # all the permutations
        x_col_surrogates = pd.DataFrame(surrogates[:, :, x_col].T, index=parcels)
        x_col_surrogates.columns = [f"surrogate_{i}" for i in range(n_perm)]
        null_distribution[:, :, x_col] = (
            pd.concat([x_col_surrogates, Y], axis=1)
            .corr(method=method)
            .iloc[: x_col_surrogates.shape[1], -Y.shape[1] :]
            .values
        )
    # step 4: calculate p value
    pvals = ((np.abs(null_distribution) >= np.abs(coefs)).sum(axis=0) + 1) / (
        n_perm + 1
    )
    # clean and return results
    coefs = coefs.reshape(Y.shape[1], X.shape[1])
    pvals = pd.DataFrame(pvals.T, index=X.columns, columns=Y.columns)
    coefs = pd.DataFrame(coefs.T, index=X.columns, columns=Y.columns)
    return coefs, pvals, null_distribution
