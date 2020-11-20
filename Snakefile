import os
import shutil
import mne
import pandas as pd
from pathlib import Path

from bids import BIDSLayout

project_root = Path()
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = project_root / 'data_bids'

layout = BIDSLayout(bids_root, validate=True)

json_files = layout.get(suffix='meg', extension='json')

subjects = [json_file.get_entities()['subject'] for json_file in json_files]
sessions = [json_file.get_entities()['session'] for json_file in json_files]

test_pipeline_dir = bids_root / 'derivatives' / 'test_pipeline'
template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg')

def find_ics(raw, ica, verbose=False):
    heart_ics, _ = ica.find_bads_ecg(raw, verbose=verbose)
    horizontal_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF14-1609', verbose=verbose)
    vertical_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF21-1609', verbose=verbose)

    all_ics = heart_ics + horizontal_eye_ics + vertical_eye_ics
    # find_bads_E*G returns list of np.int64, not int
    all_ics = map(int, all_ics)
    # Remove duplicates.
    all_ics = list(set(all_ics))

    return all_ics

def find_ics_iteratively(raw, ica, verbose=False):
    ics = []

    new_ics = True  # so that the while loop initiates at all
    while new_ics:
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        ics += new_ics

    return ics

rule all:
    input:
         expand(os.path.join(test_pipeline_dir, template+'_PSD_raw.png'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'.fif'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'_PSD_linearly_filtered.png'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'.ica'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'.ics.pickle'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'_ics_properties.png'), zip, subject=subjects,
                session=sessions),

rule linear_filtering:
    input:
        os.path.join(bids_root, template+'.json')
    output:
        os.path.join(test_pipeline_dir, template+'.fif')
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw: mne.io.ctf.ctf.RawCTF = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        raw.resample(300, npad='auto')
        raw.notch_filter([60, 120],
                 filter_length='auto',
                 phase='zero')
        raw.filter(0.3, None, fir_design='firwin')
        raw.save(output[0])

rule draw_raw_psd:
    input:
        os.path.join(bids_root, template+'.json')
    output:
        os.path.join(test_pipeline_dir, template+'_PSD_raw.png')
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)
        p.savefig(output[0])

rule draw_psd_linearly_filtered:
    input:
        os.path.join(test_pipeline_dir, template+'.fif')
    output:
        os.path.join(test_pipeline_dir, template+'_PSD_linearly_filtered.png')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)
        p.savefig(output[0])

rule fit_ica:
    input:
        os.path.join(test_pipeline_dir, template+'.fif')
    output:
        os.path.join(test_pipeline_dir, template+'.ica')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        raw.filter(1, None)
        ica = mne.preprocessing.ICA(random_state=2, n_components=25, verbose=False)
        ica.fit(raw)
        ica.save(output[0])

rule find_ics:
    input:
        os.path.join(test_pipeline_dir, template+'.fif'),
        os.path.join(test_pipeline_dir, template+'.ica')
    output:
        os.path.join(test_pipeline_dir, template+'.ics.pickle'),
        os.path.join(test_pipeline_dir, template+'_ics_properties.png')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(input[1])
        ics = find_ics_iteratively(raw, ica, verbose=False)
        pd.to_pickle(ics, output[0])
        p = ica.plot_properties(raw, picks=ics, show=False)
        p[0].savefig(output[1])

