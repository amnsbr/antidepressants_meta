import os
import sys
import copy
import pandas as pd
import numpy as np
import nilearn.image, nilearn.surface
import nimare.dataset

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

PYALE_DIR = os.path.join(PROJECT_DIR, "tools", "pyALE")
sys.path.append(PYALE_DIR)
from nb_pipeline import setup, analysis

sys.path.append(os.path.join(PROJECT_DIR, "scripts"))
from helpers import io

def merge_experiments(dset):
    """
    Merges multiple experiments (contrasts) of the same study
    into a single experiment.

    Parameters
    ----------
    dset: nimare.dataset.Dataset
        NiMARE dataset

    Returns
    -------
    dset: nimare.dataset.Dataset
        NiMARE dataset with merged experiments
    """
    # make a copy of the dataset to avoid overwriting data
    dset = copy.deepcopy(dset)
    # merge metadata
    metadata_merged = pd.DataFrame(columns=dset.metadata.columns[:6])
    for i, (study_id, study_metadata) in enumerate(dset.metadata.groupby('study_id')):
        metadata_merged.loc[i, :] = {
            'id': f'{study_id}-0',
            'study_id': study_id,
            'contrast_id': 0,
            'sample_sizes': [int(study_metadata['sample_sizes'].apply(lambda c: c[0]).mean().round())],
            'author': study_metadata.iloc[0].loc['author'],
            'year': study_metadata.iloc[0].loc['year']
        }
    # merge coordinates
    coordinates_merged = []
    for i, (study_id, study_coordinates) in enumerate(dset.coordinates.groupby('study_id')):
        curr_coordinates = study_coordinates.copy()
        curr_coordinates.loc[:, 'id'] = f'{study_id}-0'
        curr_coordinates.loc[:, 'contrast_id'] = 0
        coordinates_merged.append(curr_coordinates)
    coordinates_merged = pd.concat(coordinates_merged)
    # create annotations, images and texts dataframes
    # (probably not needed)
    annotations_merged = metadata_merged.iloc[:, :3]
    # merge IDs
    ids_merged = metadata_merged.loc[:, 'id'].unique()
    # apply the merging on the dataset and update it
    dset.metadata = metadata_merged
    dset.coordinates = coordinates_merged
    dset.annotations = annotations_merged
    ## .ids cannot be set so here we use slice
    dset = dset.slice(ids_merged)
    return dset

def nimare_to_pyale(dset):
    exps_coordinates = []
    # an empty row to put after each experiment
    empty_row = pd.DataFrame(
        np.NaN,
        index=[0], 
        columns=['Experiments', 'Subjects', 'X', 'Y', 'Z', 'Space']
    )
    for study_id in dset.metadata['study_id'].unique():
        # get sample size from metadata and merge contrasts if 
        # needed (in that case take rounded mean as sample size)
        curr_exps_metadata = dset.metadata.loc[
            dset.metadata['study_id']==study_id
        ]
        sample_size = round(curr_exps_metadata['sample_sizes'].apply(lambda c: c[0]).mean())\
        # filter study coordinates data and remove columns not needed
        curr_exps_coordinates = dset.coordinates.loc[
            dset.coordinates['study_id']==str(study_id)
        ]
        curr_exps_coordinates = curr_exps_coordinates.drop(columns=['id', 'contrast_id'])
        # add sample size
        curr_exps_coordinates['Subjects'] = sample_size
        # clean space
        ## Note: this is unnecessary since in dump all
        ## coordinates are already converted to MNI
        ## but is put here to make the code work with other
        ## sources which may contain different spaces
        if len(curr_exps_coordinates['space'].unique()) > 1:
            raise NotImplementedError(
                "Different experiments of the same study"
                " must have the same space"
            )
        space = curr_exps_coordinates['space'].iloc[0]
        space = space[:3].upper() # mni* -> MNI; tal* -> TAL
        curr_exps_coordinates['Space'] = space
        curr_exps_coordinates = curr_exps_coordinates.drop(columns=['space'])
        # rename columns
        curr_exps_coordinates = curr_exps_coordinates.rename(columns={
            'study_id': 'Experiments',
            'x': 'X',
            'y': 'Y',
            'z': 'Z'
        })
        curr_exps_coordinates['Experiments'] = 'study-' + curr_exps_coordinates['Experiments']
        # reorder columns
        curr_exps_coordinates = curr_exps_coordinates.loc[:, empty_row.columns]
        # add an empty row at the end
        curr_exps_coordinates = pd.concat([
            curr_exps_coordinates,
            empty_row
        ]).reset_index(drop=True)
        exps_coordinates.append(curr_exps_coordinates)
    macm_df = (
        pd.concat(exps_coordinates)
        .set_index('Experiments')
        .iloc[:-1] # remove last empty row
    )
    return macm_df

def run(cluster_img, analysis_name=None):
    """
    Run MACM analysis on a given cluster (binary) image

    Parameters
    ----------
    cluster_img: str
        path to the cluster image
        Note that the image should be in MNI space
        and binary (1 for the cluster, 0 for the rest)
        and it should include a single cluster
    """
    # set the output directory
    if analysis_name is None:
        analysis_name = os.path.basename(cluster_img).replace('.nii', '')
    out_dir = os.path.join(io.RESULTS_DIR, 'macm', analysis_name)
    if os.path.exists(os.path.join(out_dir, 'Results')):
        print(
            f"{analysis_name} is already done or started"
            f"\n If you wish to repeat it remove {os.path.join(out_dir, 'Results')}"
        )
        return
    os.makedirs(out_dir, exist_ok=True)
    # first create a dataset of experiments with
    # at least one reported actviation in the cluster
    ## load BrainMap dump
    dump = nimare.dataset.Dataset.load(os.path.join(
        io.SRC_DIR, 'BrainMap_dump_Feb2024_mask-Grey10.pkl.gz'
    ))
    ## create cluster mask
    cluster_mask = nilearn.image.binarize_img(cluster_img)
    ## mask dump to experiments with activations in the cluster
    seed_dset = dump.slice(dump.get_studies_by_mask(cluster_mask))
    ## merge multiple experiments of the same study
    seed_dset = merge_experiments(seed_dset)
    ## save dset
    seed_dset.save(os.path.join(out_dir, 'dset.pkl.gz'))
    # second, create excel files required by pyALE
    ## convert nimare dataset to a excel file
    macm_df = nimare_to_pyale(seed_dset)
    ## create the analysis excel file
    macm_analysis = pd.DataFrame([{
        0: 'M',
        1: 'MACM',
        2: '+ALL'
    }]).set_index(0)
    ## save them
    macm_df.to_excel(os.path.join(out_dir, 'coordinates.xlsx'))
    macm_analysis.to_excel(os.path.join(out_dir, 'analysis.xlsx'))
    # last, run an ALE meta-analysis using pyALE
    meta_df, exp_all, tasks = setup(out_dir, 'analysis.xlsx', 'coordinates.xlsx')
    
    analysis(path = out_dir,
             meta_df = meta_df,
             exp_all = exp_all,
             tasks = tasks,
             tfce_enabled = True,
             null_repeats = 10000,
             cluster_thresh = 0.001,
             sample_n = 2500,
             nprocesses = 6)

if __name__ == '__main__':
    cluster_img = os.path.join(io.RESULTS_DIR, 'ale', 'Contrast', 
                    'Results', 'MainEffect', 'Full', 'Volumes', 'Corrected',
                    'Contrast_increased_cFWE05.nii')
    run(cluster_img)