import os
import shutil
import mne
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

rule all:
    input:
         expand(os.path.join(test_pipeline_dir, template+'_PSD_raw.png'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'.fif'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'_PSD_linearly_filtered.png'), zip, subject=subjects,
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
