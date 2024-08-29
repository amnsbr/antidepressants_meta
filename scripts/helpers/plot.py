import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nibabel
import nilearn.datasets
import neuromaps.datasets
import brainspace.mesh
import brainspace.plotting
import pyvirtualdisplay
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm


PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC_DIR = os.path.join(PROJECT_DIR, "scripts", "src")
MESHES_FSLR = {
    "L": str(neuromaps.datasets.fetch_fslr()["inflated"].L),
    "R": str(neuromaps.datasets.fetch_fslr()["inflated"].R),
}

from . import transform

# order of subcortex structures expected by
# engimatoolbox (excluding ventricles)
SUBCORTEX_ORDER = [
    "Left Accumbens",
    "Left Amygdala",
    "Left Caudate",
    "Left Hippocampus",
    "Left Pallidum",
    "Left Putamen",
    "Left Thalamus",
    "Right Accumbens",
    "Right Amygdala",
    "Right Caudate",
    "Right Hippocampus",
    "Right Pallidum",
    "Right Putamen",
    "Right Thalamus",
]


# create seismic light in the 20% to 80% range of seismic
seismic = plt.get_cmap("seismic")
seismic_light = LinearSegmentedColormap.from_list(
    "seismic_light", [seismic(x) for x in np.linspace(0.2, 0.8, 256)]
)
# register it
if not "seismic_light" in matplotlib.colormaps():
    matplotlib.cm.register_cmap(name="seismic_light", cmap=seismic_light)


def plot_surface(
    surface_data,
    mesh=MESHES_FSLR,
    itype=None,
    filename=None,
    layout_style="row",
    cmap="viridis",
    vrange=None,
    cbar=False,
    nan_color=(0.75, 0.75, 0.75, 1),
    **plotter_kwargs,
):
    """
    Plots `surface_data` on `mesh` using brainspace

    Parameters
    ----------
    surface_data: (np.ndarray)
    mesh: (dict)
        path to meshes for 'L' and 'R'
    itype: (str | None)
        mesh file type. For .gii enter None. For freesurfer files enter 'fs'
    filename: (Pathlike str)
    layout_style: (str)
        - row
        - grid
    cmap: (str)
    vrange: (tuple | None)
    nan_color: (tuple)
    **plotter_kwargs
    """
    # create virtual display for plotting in remote servers
    disp = pyvirtualdisplay.Display(visible=False)
    disp.start()
    # load surface mesh
    if isinstance(mesh, str):
        if mesh in ["fsaverage", "fsaverage5"]:
            mesh = {
                "L": nilearn.datasets.fetch_surf_fsaverage(mesh)["infl_left"],
                "R": nilearn.datasets.fetch_surf_fsaverage(mesh)["infl_right"],
            }
            itype = None
        else:
            raise ValueError("Unknown mesh")
    else:
        for fs_suffix in [".pial", ".midthickness", ".white", ".inflated"]:
            if mesh["L"].endswith(fs_suffix):
                itype = "fs"
    if not os.path.exists(mesh["L"]):
        raise ValueError("Mesh not found")
    lh_surf = brainspace.mesh.mesh_io.read_surface(mesh["L"], itype=itype)
    rh_surf = brainspace.mesh.mesh_io.read_surface(mesh["R"], itype=itype)
    # configurations
    if filename:
        screenshot = True
        embed_nb = False
        filename += ".png"
    else:
        screenshot = False
        embed_nb = True
    if layout_style == "row":
        size = (1600, 400)
        zoom = 1.2
    else:
        size = (900, 500)
        zoom = 1.8
    if vrange is None:
        vrange = (np.nanmin(surface_data), np.nanmax(surface_data))
    elif vrange == "sym":
        vmin = min(np.nanmin(surface_data), -np.nanmax(surface_data))
        vrange = (vmin, -vmin)
    if cbar:
        plot_colorbar(vrange[0], vrange[1], cmap)
    return brainspace.plotting.surface_plotting.plot_hemispheres(
        lh_surf,
        rh_surf,
        surface_data,
        layout_style=layout_style,
        cmap=cmap,
        color_range=vrange,
        size=size,
        zoom=zoom,
        interactive=False,
        embed_nb=embed_nb,
        screenshot=screenshot,
        filename=filename,
        transparent_bg=True,
        nan_color=nan_color,
        **plotter_kwargs,
    )


def plot_parc_subcortical(
    parc_data,
    schaefer=400,
    tian="S2",
    nan_zero=True,
    color_range="whole-brain",
    zoom=1.3,
    size=(1200, 300),
    **plotter_kwargs,
):
    """
    Plots subcortical part of data parcellated using Schaefer-Tian
    parcellations as a mesh

    Parameters
    ----------
    parc_data: (pd.DataFrame)
    schaefer: (int)
        Schaefer parcellation number of parcels
    tian: {'S1', 'S2'}
        Tian parcellation granularity
    nan_zero: (bool)
        Set NaNs to zero
    color_range: {'whole-brain', tuple}
        - 'whole-brain': sets it to whole-brain min and max
        - tuple: custom ranges
    zoom: (int)
        Plot zoom
    size: (tuple)
        Plot size
    **plotter_kwargs
        Keyword arguments to `enigmatoolbox.plotting.surface_plotting.plot_subcortical`
    """
    # create virtual display for plotting in remote servers
    disp = pyvirtualdisplay.Display(visible=False)
    disp.start()
    # get translation map from tian to the 14 structures expected
    # by enigmatoolbox plotter
    atlas_cifti = nibabel.load(
        os.path.join(
            SRC_DIR,
            f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii",
        )
    )
    labels = [
        l.label
        for i, l in atlas_cifti.header.get_index_map(0)._maps[0].label_table.items()
    ][1:]
    subcortical_labels = np.where(~pd.Series(labels).str.contains("7Networks"))
    subcortical_labels = np.array(labels)[subcortical_labels]
    tian_to_mesh = pd.Series(index=subcortical_labels)
    abbr_to_full = {
        "HIP": "Hippocampus",
        "AMY": "Amygdala",
        "THA": "Thalamus",
        "NAc": "Accumbens",
        "GP": "Pallidum",
        "PUT": "Putamen",
        "CAU": "Caudate",
    }
    for label in subcortical_labels:
        hem = "Left" if "-lh" in label else "Right"
        for key in abbr_to_full:
            if key in label:
                tian_to_mesh.loc[label] = f"{hem} {abbr_to_full[key]}"
    plot_data = pd.DataFrame(
        {"data": parc_data.loc[subcortical_labels], "mesh_name": tian_to_mesh}
    )
    plot_data = plot_data.groupby("mesh_name").mean().loc[SUBCORTEX_ORDER]
    if nan_zero:
        plot_data[plot_data == 0] = (
            np.NaN
        )  # this makes the plot nicer, matching the cortical plot
    # if indicated match color range of the whole brain
    if color_range == "whole-brain":
        color_range = (np.nanmin(parc_data.values), np.nanmax(parc_data.values))
    # plot
    return plot_subcortical(
        plot_data.values.flatten(),
        ventricles=False,
        size=size,
        zoom=zoom,
        color_range=color_range,
        color_bar=False,
        embed_nb=True,
        **plotter_kwargs,
    )


def plot_cifti_subcortical(
    cifti_data,
    nan_zero=True,
    color_range="whole-brain",
    zoom=1.3,
    size=(1200, 300),
    **plotter_kwargs,
):
    """
    Plots subcortical part of cifti data

    Parameters
    ----------
    cifti_data: (np.ndarray)
        Data in HCP Cifti space
    nan_zero: (bool)
        Set NaNs to zero
    color_range: {'whole-brain', tuple}
        - 'whole-brain': sets it to whole-brain min and max
        - tuple: custom ranges
    zoom: (int)
        Plot zoom
    size: (tuple)
        Plot size
    **plotter_kwargs
        Keyword arguments to `enigmatoolbox.plotting.surface_plotting.plot_subcortical`
    """
    # create virtual display for plotting in remote servers
    disp = pyvirtualdisplay.Display(visible=False)
    disp.start()
    # set the subcortical parcels order to the order
    # expectec by enigmatoolbox
    plot_data = []
    for subcortical_struc in SUBCORTEX_ORDER:
        plot_data.append(
            np.nanmean(cifti_data[transform.CIFTI_INDICES[subcortical_struc]])
        )
    plot_data = np.array(plot_data)
    if nan_zero:
        plot_data[plot_data == 0] = np.NaN
    # if indicated match color range of the whole brain
    if color_range == "whole-brain":
        color_range = (np.nanmin(cifti_data), np.nanmax(cifti_data))
    # plot
    return plot_subcortical(
        plot_data.flatten(),
        ventricles=False,
        size=size,
        zoom=zoom,
        color_range=color_range,
        color_bar=False,
        embed_nb=True,
        **plotter_kwargs,
    )


def plot_colorbar(
    vmin, vmax, cmap=None, bins=None, orientation="vertical", figsize=None
):
    """
    Plots a colorbar

    Parameters
    ---------
    vmin, vmax: (float)
    cmap: (str or `matplotlib.colors.Colormap`)
    bins: (int)
        if specified will plot a categorical cmap
    orientation: (str)
        - 'vertical'
        - 'horizontal'
    figsize: (tuple)

    Returns
    -------
    fig: (`matplotlib.figure.Figure`)
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        np.linspace(vmin, vmax, 100).reshape(10, 10), cmap=plt.cm.get_cmap(cmap, bins)
    )
    fig.gca().set_visible(False)
    divider = make_axes_locatable(ax)
    if orientation == "horizontal":
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
    else:
        cax = divider.append_axes("left", size="10%", pad=0.05)
    fig.colorbar(im, cax=cax, ticks=np.array([vmin, vmax]), orientation=orientation)
    cax.yaxis.tick_left()
    cax.xaxis.tick_bottom()
    return fig


"""
Below selected functions from enigmatoolbox are included which
are used for subcortical plotting. Enigmatoolbox itself was not
possible to use due to discrepency between working versions of vtk,
brainspace and enigmatoolbox.

The code is copied from https://github.com/MICA-MNI/ENIGMA/tree/465eac00de3b1015b81a7e90c23a77237821968b
and very slightly adapted
"""


def subcorticalvertices(subcortical_values=None):
    """
    Map one value per subcortical area to surface vertices (author: @saratheriver)

    Parameters
    ----------
    subcortical_values : 1D ndarray
        Shape = (16,), order of subcortical structure must be = L_accumbens, L_amygdala, L_caudate, L_hippocampus,
        L_pallidun, L_putamen, L_thalamus, L_ventricles, R_accumbens, R_amygdala, R_caudate, R_hippocampus,
        R_pallidun, R_putamen, R_thalamus, R_ventricles

    Returns
    -------
    data : 1D ndarray
        Transformed data, shape = (51278,)
    """
    numvertices = [
        867,
        1419,
        3012,
        3784,
        1446,
        4003,
        3726,
        7653,
        838,
        1457,
        3208,
        3742,
        1373,
        3871,
        3699,
        7180,
    ]
    data = []
    if isinstance(subcortical_values, np.ndarray):
        for ii in range(16):
            data.append(np.tile(subcortical_values[ii], (numvertices[ii], 1)))
        data = np.vstack(data).flatten()
    return data


def plot_subcortical(
    array_name=None,
    ventricles=True,
    color_bar=False,
    color_range=None,
    label_text=None,
    cmap="RdBu_r",
    nan_color=(1, 1, 1, 0),
    zoom=1,
    background=(1, 1, 1),
    size=(400, 400),
    interactive=True,
    embed_nb=False,
    screenshot=False,
    filename=None,
    scale=(1, 1),
    transparent_bg=True,
    **kwargs,
):
    """
    Plot subcortical surface with lateral and medial views (author: @saratheriver)

    Parameters
    ----------
    array_name : str, list of str, ndarray or list of ndarray, optional
        Name of point data array to plot. If ndarray, the array is split for
        the left and right hemispheres. If list, plot one row per array.
        Default is None.
    ventricles : bool, optional
        Whether to include ventricles (i.e., array_name must have 16 values).
        False does not include ventricles (e.g., array_name must have 14 values).
        Default is True.
    color_bar : bool, optional
        Plot color bar for each array (row). Default is False.
    color_range : {'sym'}, tuple or sequence.
        Range for each array name. If 'sym', uses a symmetric range. Only used
        if array has positive and negative values. Default is None.
    label_text : dict[str, array-like], optional
        Label text for column/row. Possible keys are {'left', 'right',
        'top', 'bottom'}, which indicate the location. Default is None.
    nan_color : tuple
        Color for nan values. Default is (1, 1, 1, 0).
    zoom : float or sequence of float, optional
        Zoom applied to the surfaces in each layout entry.
    background : tuple
        Background color. Default is (1, 1, 1).
    cmap : str, optional
        Color map name (from matplotlib). Default is 'RdBu_r'.
    size : tuple, optional
        Window size. Default is (400, 400).
    interactive : bool, optional
        Whether to enable interaction. Default is True.
    embed_nb : bool, optional
        Whether to embed figure in notebook. Only used if running in a
        notebook. Default is False.
    screenshot : bool, optional
        Take a screenshot instead of rendering. Default is False.
    filename : str, optional
        Filename to save the screenshot. Default is None.
    transparent_bg : bool, optional
        Whether to us a transparent background. Only used if
        ``screenshot==True``. Default is False.
    scale : tuple, optional
        Scale (magnification). Only used if ``screenshot==True``.
        Default is None.
    kwargs : keyword-valued args
        Additional arguments passed to the plotter.

    Returns
    -------
    figure : Ipython Image or None
        Figure to plot. None if using vtk for rendering (i.e.,
        ``embed_nb == False``).

    See Also
    --------
    :func:`build_plotter`
    :func:`plot_surf`
    """
    if color_bar is True:
        color_bar = "right"

    surf_lh = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, "sctx_lh.gii"))
    surf_rh = brainspace.mesh.mesh_io.read_surface(os.path.join(SRC_DIR, "sctx_rh.gii"))

    surfs = {"lh": surf_lh, "rh": surf_rh}
    layout = ["lh", "lh", "rh", "rh"]
    view = ["lateral", "medial", "lateral", "medial"]

    if isinstance(array_name, pd.Series):
        array_name = array_name.to_numpy()

    if array_name.shape == (1, 16) or array_name.shape == (1, 14):
        array_name = np.transpose(array_name)

    if len(array_name) == 16 and ventricles:
        array_name = subcorticalvertices(array_name)
    elif len(array_name) == 14 and ventricles is False:
        array_name3 = np.empty(16)
        array_name3[:] = np.nan
        array_name3[0:7] = array_name[0:7]
        array_name3[8:15] = array_name[7:]
        array_name = subcorticalvertices(array_name3)

    if isinstance(array_name, np.ndarray):
        if array_name.ndim == 2:
            array_name = [a for a in array_name]
        elif array_name.ndim == 1:
            array_name = [array_name]

    if isinstance(array_name, list):
        layout = [layout] * len(array_name)
        array_name2 = []
        n_pts_lh = surf_lh.n_points
        for an in array_name:
            if isinstance(an, np.ndarray):
                name = surf_lh.append_array(an[:n_pts_lh], at="p")
                surf_rh.append_array(an[n_pts_lh:], name=name, at="p")
                array_name2.append(name)
            else:
                array_name2.append(an)
        array_name = np.asarray(array_name2)[:, None]

    if isinstance(cmap, list):
        cmap = np.asarray(cmap)[:, None]

    kwds = {"view": view, "share": "r"}
    kwds.update(kwargs)
    return brainspace.plotting.surface_plotting.plot_surf(
        surfs,
        layout,
        array_name=array_name,
        color_bar=color_bar,
        color_range=color_range,
        label_text=label_text,
        cmap=cmap,
        nan_color=nan_color,
        zoom=zoom,
        background=background,
        size=size,
        interactive=interactive,
        embed_nb=embed_nb,
        screenshot=screenshot,
        filename=filename,
        scale=scale,
        transparent_bg=transparent_bg,
        **kwds,
    )
