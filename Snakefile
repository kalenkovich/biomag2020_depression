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

rule all:
    input:
         expand(os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg',
                             'sub-{subject}_ses-{session}_task-restingstate_meg.json'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg',
                             'sub-{subject}_ses-{session}_task-restingstate_meg_PSD_raw.png'), zip, subject=subjects,
                session=sessions)

rule copy_json:
    input:
        os.path.join(bids_root, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json')
    output:
        os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json')
    run:
         shutil.copy(input[0], output[0])

rule draw_raw_psd:
    input:
        os.path.join(bids_root, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json')
    output:
        os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg_PSD_raw.png')
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        raw.plot_psd(show=False).savefig(output[0])
