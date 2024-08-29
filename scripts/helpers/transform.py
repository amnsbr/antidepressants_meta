import os
import types
import numpy as np
import pandas as pd
import nibabel
import neuromaps.datasets
import nilearn.image
from nilearn.maskers import NiftiLabelsMasker

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC_DIR = os.path.join(PROJECT_DIR, "scripts", "src")
HCP_DIR = os.environ.get("HCP_DIR")


def get_cifti_brainmodelaxis_data():
    """
    Gets the brainmodelaxis data of the HCP CIFTI
    
    Returns
    -------
    cifti_brainmodelaxis: (types.SimpleNamespace)
        voxel: (np.ndarray) ijk coordinates of voxels
        vertex: (np.ndarray) fsLR vertex indices of the surface cortical vertices
        volume_mask: (np.ndarray) mask of volumetric grayordinates
        surface_mask: (np.ndarray) mask of surface grayordinates
        affine: (np.ndarray) affine
        volume_shape: (tuple) volume dimensions
    """
    out_path = os.path.join(SRC_DIR, "HCP_brainmodelaxis_data.npz")
    if os.path.exists(out_path):
        out_data = dict(np.load(out_path))
        return types.SimpleNamespace(**out_data)
    assert HCP_DIR is not None
    # load source CIFTI image
    # this is an example MNI image in CIFTI format
    # based on '100206/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_stats.dscalar.nii'
    # The subject does not matter here because we're only using the brainmodelaxis
    # which includes the coordinates and indices of subcortical voxels and indices
    # of cortical vertices which are needed for conversion between HCP CIFTI and
    # fsLR / MNI
    cifti_mni = nibabel.load(
        os.path.join(
            HCP_DIR,
            "100206",
            "MNINonLinear",
            "Results",
            "rfMRI_REST1_LR",
            "rfMRI_REST1_LR_Atlas_stats.dscalar.nii",
        )
    )
    cifti_brainmodelaxis = cifti_mni.header.get_axis(1)
    # select the components needed
    out_data = {
        # ijk coordinates of voxels
        "voxel": cifti_brainmodelaxis.voxel,
        # fsLR vertex indices of the surface cortical vertices
        # Note: Cifti vertices exclude midline, but fsLR has them
        # this helps for mapping between the two
        "vertex": cifti_brainmodelaxis.vertex,
        # mask of volumetric grayordinates
        "volume_mask": cifti_brainmodelaxis.volume_mask,
        # mask of surface grayordinates
        "surface_mask": cifti_brainmodelaxis.surface_mask,
        # affine
        "affine": cifti_brainmodelaxis.affine,
        # volume dimensions
        "volume_shape": cifti_brainmodelaxis.volume_shape,
    }
    # save it
    np.savez_compressed(out_path, **out_data)
    return types.SimpleNamespace(**out_data)


CIFTI_NPOINTS = 91282
FSLR_N_VERT_HEM = 32492
CIFTI_BRAINMODELAXIS = get_cifti_brainmodelaxis_data()
# structure to CIFTI index mapping based on https://github.com/rmldj/hcp-utils
CIFTI_INDICES = {
    "Left Cortex": slice(0, 29696, None),
    "Right Cortex": slice(29696, 59412, None),
    "Cortex": slice(0, 59412, None),
    "Subcortex": slice(59412, None, None),
    "Left Accumbens": slice(59412, 59547, None),
    "Right Accumbens": slice(59547, 59687, None),
    "Left Amygdala": slice(59687, 60002, None),
    "Right Amygdala": slice(60002, 60334, None),
    "Brainstem": slice(60334, 63806, None),
    "Left Caudate": slice(63806, 64534, None),
    "Right Caudate": slice(64534, 65289, None),
    "Left Cerebellum": slice(65289, 73998, None),
    "Right Cerebellum": slice(73998, 83142, None),
    "Left Diencephalon": slice(83142, 83848, None),
    "Right Diencephalon": slice(83848, 84560, None),
    "Left Hippocampus": slice(84560, 85324, None),
    "Right Hippocampus": slice(85324, 86119, None),
    "Left Pallidum": slice(86119, 86416, None),
    "Right Pallidum": slice(86416, 86676, None),
    "Left Putamen": slice(86676, 87736, None),
    "Right Putamen": slice(87736, 88746, None),
    "Left Thalamus": slice(88746, 90034, None),
    "Right Thalamus": slice(90034, None, None),
}
N_VERTICES_HEM_FSLR = 32492
FSLR_CORTEX = np.zeros(N_VERTICES_HEM_FSLR * 2, dtype=bool)
FSLR_CORTEX[
    np.concatenate(
        [
            CIFTI_BRAINMODELAXIS.vertex[CIFTI_INDICES["Left Cortex"]],
            CIFTI_BRAINMODELAXIS.vertex[CIFTI_INDICES["Right Cortex"]]
            + N_VERTICES_HEM_FSLR,
        ]
    )
] = True


def get_cifti_mni_coordinates():
    """
    Gets MNI xyz coordinates corresponding to each grayordinate
    of the standard HCP CIFTI image

    Returns
    -------
    all_xyz: (np.ndarray)
        Shape (91282, 3)
    """
    out_path = os.path.join(SRC_DIR, "MNINonLinear_CIFTI_coords.txt")
    if os.path.exists(out_path):
        return np.loadtxt(out_path)
    # get the subcortical voxel indices
    subcortical_ijk = CIFTI_BRAINMODELAXIS.voxel[CIFTI_BRAINMODELAXIS.volume_mask]
    # transform voxel indices to xyz coordinates
    subcortical_xyz = nibabel.affines.apply_affine(
        CIFTI_BRAINMODELAXIS.affine, subcortical_ijk
    )
    # cortex
    # load MNI152 coordinates of all surface fsLR vertices from neuromaps regfusion
    regfusion_paths = neuromaps.datasets.fetch_regfusion("fsLR")["32k"]
    fslr_surf_coords = {
        "L": np.loadtxt(regfusion_paths.L),
        "R": np.loadtxt(regfusion_paths.R),
    }
    # select vertices that exist in the cortex
    valid_vertices = CIFTI_BRAINMODELAXIS.vertex[CIFTI_BRAINMODELAXIS.surface_mask]
    valid_vertices_hem = {
        "L": valid_vertices[CIFTI_INDICES["Left Cortex"]],
        "R": valid_vertices[CIFTI_INDICES["Right Cortex"]],
    }
    cortical_xyz = np.concatenate(
        [
            fslr_surf_coords["L"][valid_vertices_hem["L"]],
            fslr_surf_coords["R"][valid_vertices_hem["R"]],
        ]
    )
    # combine cortex and subcortex
    all_xyz = np.concatenate([cortical_xyz, subcortical_xyz])
    np.savetxt(out_path, all_xyz)
    return all_xyz


def cifti_to_fsLR(cifti_data, concat=True):
    """
    Transforms data from HCP CIFTI space to the correct fsLR space

    Parameters
    ----------
    cifit_data: (np.ndarray)
        Shape (k, 91282) or (91282,)
    concat: (bool)
    
    Returns
    -------
    fsLR_data: (np.ndarray)
        Shape (k, 64984) or (64984,)
    or
    fsLR_data: (dict)
        L: (np.ndarray) Shape (k, 32492) or (32492,)
        R: (np.ndarray) Shape (k, 32492) or (32492,)
    """
    cort_vertices = CIFTI_BRAINMODELAXIS.vertex[CIFTI_BRAINMODELAXIS.surface_mask]
    cort_vertices_hem = {
        "L": cort_vertices[CIFTI_INDICES["Left Cortex"]],
        "R": cort_vertices[CIFTI_INDICES["Right Cortex"]],
    }
    # transform Cifti within-cortical data to
    # fsLR (which basically adds NaNs of the midline
    # in correct indices)
    cifti_data = np.atleast_2d(cifti_data)
    lh_fsLR = np.zeros((cifti_data.shape[0], N_VERTICES_HEM_FSLR)) * np.NaN
    lh_fsLR[:, cort_vertices_hem["L"]] = cifti_data[:, CIFTI_INDICES["Left Cortex"]]
    rh_fsLR = np.zeros((cifti_data.shape[0], N_VERTICES_HEM_FSLR)) * np.NaN
    rh_fsLR[:, cort_vertices_hem["R"]] = cifti_data[:, CIFTI_INDICES["Right Cortex"]]
    if concat:
        return np.concatenate([lh_fsLR, rh_fsLR], axis=1).squeeze()
    else:
        return {"L": lh_fsLR.squeeze(), "R": rh_fsLR.squeeze()}


def fsLR_to_cifti(fslr_data):
    """
    Converts fsLR data to an array with HCP Cifti shape
    which has zeros for subcortex

    Parameters
    ----------
    fslr_data: (np.ndarray)
        Shape (k, 64984) or (64984,)
    
    Returns
    -------
    cifti_data: (np.ndarray)
        Shape (k, 91282) or (91282,)
    """
    fslr_data = np.atleast_2d(fslr_data)
    cifti_data = np.zeros((fslr_data.shape[0], CIFTI_NPOINTS)) * np.NaN
    cort_vertices = CIFTI_BRAINMODELAXIS.vertex[CIFTI_BRAINMODELAXIS.surface_mask]
    cort_vertices_hem = {
        "L": cort_vertices[CIFTI_INDICES["Left Cortex"]],
        "R": cort_vertices[CIFTI_INDICES["Right Cortex"]],
    }
    fslr_data_hem = {
        "L": fslr_data[:, :N_VERTICES_HEM_FSLR],
        "R": fslr_data[:, N_VERTICES_HEM_FSLR:],
    }
    cifti_data[:, CIFTI_INDICES["Left Cortex"]] = fslr_data_hem["L"][
        :, cort_vertices_hem["L"]
    ]
    cifti_data[:, CIFTI_INDICES["Right Cortex"]] = fslr_data_hem["R"][
        :, cort_vertices_hem["R"]
    ]
    return cifti_data


def cifti_to_mni(cifti_data, exc_cortex=False, exc_subcortex=False):
    """
    Maps HCP CIFTI data to MNI space

    Parameters
    ----------
    cifti_data: (np.ndarray)
        Shape (k, 91282) or (91282,)
    exc_cortex: (bool)
    exc_subcortex: (bool)

    Returns
    -------
    mni: Nifti image
    """
    # load MNI 2mm
    mni_mask = nilearn.datasets.load_mni152_brain_mask(2)
    # get ijk of cifti grayordinate MNI coordinates
    # in this space
    all_xyz = get_cifti_mni_coordinates()
    all_ijk = nibabel.affines.apply_affine(
        np.linalg.inv(mni_mask.affine), all_xyz
    ).astype("int")
    # exclude cortex or subcortex if indicated
    _cifti_data = cifti_data.copy()
    if exc_cortex:
        all_ijk = all_ijk[CIFTI_BRAINMODELAXIS.volume_mask]
        _cifti_data = _cifti_data[CIFTI_BRAINMODELAXIS.volume_mask]
    elif exc_subcortex:
        all_ijk = all_ijk[CIFTI_BRAINMODELAXIS.surface_mask]
        _cifti_data = _cifti_data[CIFTI_BRAINMODELAXIS.surface_mask]
    # initialize MNI data as NaNs
    mni_data = np.zeros(mni_mask.shape) * np.NaN
    # map from grayordinate to voxel space
    mni_data[all_ijk[:, 0], all_ijk[:, 1], all_ijk[:, 2]] = _cifti_data
    # write the data to the cifti mni
    mni = nilearn.image.new_img_like(mni_mask, mni_data)
    return mni


def parcellate_volumetric(img_path, schaefer="400", tian="S2", nonzero=False):
    """
    Parcellated volumetric image

    Parameters
    ----------
    img_path: (str)
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    nonzero: (bool)
        calculate the average only among non-zero voxels
    
    Returns
    -------
    parcellated_data: (pd.Series)
        Shape (n_parc,)
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
    parc_img = os.path.join(SRC_DIR, parc_filename_mni)
    if nonzero:
        masker = NiftiLabelsMasker(
            parc_img, strategy="sum", resampling_target="data", background_label=0
        )
        # >> count the number of non-zero voxels per parcel so the average
        # is calculated only among non-zero voxels
        nonzero_mask = nilearn.image.math_img("img != 0", img=img_path)
        nonzero_voxels_count_per_parcel = masker.fit_transform(nonzero_mask).flatten()
        # >> take the average of PET values across non-zero voxels
        img_parcel_sum = masker.fit_transform(img_path).flatten()
        parcellated_data = img_parcel_sum / nonzero_voxels_count_per_parcel
    else:
        masker = NiftiLabelsMasker(
            parc_img, strategy="mean", resampling_target="data", background_label=0
        )
        parcellated_data = masker.fit_transform(img_path).flatten()
    # get labels excluding background
    atlas_cifti = nibabel.load(os.path.join(SRC_DIR, parc_filename_cifti))
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ][1:]
    # add labels of the parcels
    parcellated_data = pd.Series(parcellated_data, index=labels)
    return parcellated_data


def parcellate_cifti(cifti_data, schaefer="400", tian="S2", drop_midline=True):
    """
    Parcellate HCP Cifti

    Parameters
    ----------
    cifti_data: (np.ndarray)
        Shape (k, 91282) or (91282,)
    schaefer: {'400'}
    tian: {'S1', 'S2', None}
    drop_midline: (bool)

    Returns
    -------
    parc_data: (pd.DataFrame)
        Shape (n_parc, k)
    """
    if tian is None:
        parc_filename_cifti = (
            f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"
        )
    else:
        parc_filename_cifti = f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii"
    # load parcellation map as a labeled array of (91282,)
    atlas_cifti = nibabel.load(os.path.join(SRC_DIR, parc_filename_cifti))
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ]
    labels_transdict = dict(enumerate(labels))
    parc = atlas_cifti.get_fdata()
    parc_labeled = np.vectorize(labels_transdict.get)(parc).squeeze()
    # reshape cifti data to (k,91282)
    cifti_data = np.atleast_2d(cifti_data)
    assert cifti_data.shape[1] == CIFTI_NPOINTS
    # parcellate data
    parc_data = (
        pd.DataFrame(cifti_data.T, index=parc_labeled)
        .reset_index(drop=False)
        .groupby("index")
        .mean()
        .loc[labels]  # reorder into original order
    )
    if drop_midline:
        parc_data = parc_data.drop(index="???")
    return parc_data


def parc_to_cifti(parc_data, schaefer="400", tian="S2"):
    """
    Project parcellated data to HCP Cifti space

    Parameters
    ----------
    parc_data: (pd.DataFrame)
        Shape (n_parc, k)
    schaefer: {'400'}
    tian: {'S1', 'S2'}

    Returns
    -------
    cifti_data: (np.ndarray)
        Shape (91282, k)
    """
    # load parcellation map as a labeled array of (91282,)
    if tian is None:
        parc_filename_cifti = (
            f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"
        )
    else:
        parc_filename_cifti = f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii"
    atlas_cifti = nibabel.load(os.path.join(SRC_DIR, parc_filename_cifti))
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ]
    labels_transdict = dict(enumerate(labels))
    parc = atlas_cifti.get_fdata()
    parc_labeled = np.vectorize(labels_transdict.get)(parc).squeeze()
    # set midline to NaNs
    parc_data.loc["???"] = np.NaN
    return parc_data.loc[parc_labeled].values
